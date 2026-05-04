#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Render normal maps of the reconstructed hair mesh from each camera viewpoint.

Reads the TOML config to find the mesh, cameras, and output directory.
Uses pyrender (OpenGL) for triangle rasterization with z-buffering.

Usage:
    pip install trimesh pyrender numpy pillow
    # On Python < 3.11 also install tomli: pip install tomli

    # Render all cameras (camera-space normals, default)
    python scripts/visualize.py configs/5067077.toml

    # World-space normals
    python scripts/visualize.py configs/5067077.toml --frame world

    # Render specific cameras
    python scripts/visualize.py configs/5067077.toml --cam_ids 400262 400300
"""

import argparse
import json
import os

import numpy as np
import pyrender
import trimesh
from PIL import Image

try:
    import tomllib
except ImportError:  # Python < 3.11
    import tomli as tomllib


def parse_args():
    p = argparse.ArgumentParser(description="Render normal maps from hair mesh")
    p.add_argument("config", help="Path to TOML config file")
    p.add_argument(
        "--cam_ids",
        nargs="*",
        default=None,
        help="Specific camera IDs to render (default: all)",
    )
    p.add_argument(
        "--chunk_vertices",
        type=int,
        default=5_000_000,
        help="Maximum vertices per render chunk",
    )
    p.add_argument(
        "--frame",
        choices=["camera", "world"],
        default="camera",
        help="Coordinate frame for the encoded normals (default: camera)",
    )
    return p.parse_args()


def load_toml(path):
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_cameras(cameras_json):
    with open(cameras_json, "r") as f:
        data = json.load(f)

    cameras = {}
    for cam_id, cam in data["cameras"].items():
        K = np.array(cam["intrinsic"], dtype=np.float64)
        E = np.array(cam["extrinsic"], dtype=np.float64)
        cameras[cam_id] = {"K": K, "E": E}
    return cameras


def read_ply_header_counts(path):
    vertex_count = None
    face_count = None
    with open(path, "rb") as f:
        for raw_line in f:
            line = raw_line.decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
            elif line.startswith("element face "):
                face_count = int(line.split()[-1])
            elif line == "end_header":
                break
    return vertex_count, face_count


def load_mesh(path):
    loaded = trimesh.load(path, process=False, maintain_order=True)
    if isinstance(loaded, trimesh.Scene):
        geometries = [
            g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)
        ]
        if not geometries:
            raise ValueError(f"No mesh geometry found in {path}")
        loaded = trimesh.util.concatenate(geometries)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Expected Trimesh, got {type(loaded).__name__}")
    return loaded


def build_render_chunks(tm, max_vertices):
    if max_vertices < 3:
        raise ValueError("--chunk_vertices must be at least 3")

    vertices = np.asarray(tm.vertices, dtype=np.float32)
    normals = np.asarray(tm.vertex_normals, dtype=np.float32)
    faces = np.asarray(tm.faces, dtype=np.int64)

    chunks = []
    start = 0
    max_faces = max_vertices * 2

    while start < len(faces):
        end = min(len(faces), start + max_faces)

        while True:
            chunk_faces = faces[start:end]
            unique_vertices, inverse = np.unique(
                chunk_faces.reshape(-1), return_inverse=True
            )
            if len(unique_vertices) <= max_vertices or end - start <= 1:
                break
            end = start + max(1, (end - start) // 2)

        chunks.append(
            {
                "vertices": vertices[unique_vertices],
                "normals": normals[unique_vertices],
                "faces": inverse.reshape((-1, 3)).astype(np.uint32, copy=False),
            }
        )
        start = end

    max_chunk_vertices = max(len(chunk["vertices"]) for chunk in chunks)
    max_chunk_faces = max(len(chunk["faces"]) for chunk in chunks)
    print(
        f"  Render chunks: {len(chunks)} "
        f"(max {max_chunk_vertices:,} vertices, {max_chunk_faces:,} faces)"
    )
    return chunks


def extrinsic_to_gl_pose(E):
    """Convert 3x4 world-to-camera [R|t] to 4x4 OpenGL camera pose (camera-to-world)."""
    R = E[:3, :3]
    t = E[:3, 3]

    cam_to_world = np.eye(4)
    cam_to_world[:3, :3] = R.T
    cam_to_world[:3, 3] = -R.T @ t

    # OpenGL: camera looks down -Z with Y up. CV: +Z with Y down.
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    return cam_to_world @ flip


def report_gl_backend(renderer):
    """Print whether pyrender is using a real GPU or a software (CPU) rasterizer."""
    try:
        from OpenGL.GL import GL_RENDERER, GL_VENDOR, glGetString

        if hasattr(renderer, "_platform") and hasattr(renderer._platform, "make_current"):
            renderer._platform.make_current()
        vendor = glGetString(GL_VENDOR)
        gl_renderer = glGetString(GL_RENDERER)
        vendor = vendor.decode("utf-8", "ignore") if vendor else ""
        gl_renderer = gl_renderer.decode("utf-8", "ignore") if gl_renderer else ""
    except Exception as e:
        plat = os.environ.get("PYOPENGL_PLATFORM", "default")
        print(f"  GL backend: unknown ({type(e).__name__}; PYOPENGL_PLATFORM={plat})")
        return

    haystack = (vendor + " " + gl_renderer).lower()
    software_markers = ("llvmpipe", "softpipe", "swrast", "osmesa", "software")
    backend = "CPU (software)" if any(m in haystack for m in software_markers) else "GPU"
    print(f"  GL backend: {backend} — {gl_renderer or '?'} (vendor: {vendor or '?'})")


def main():
    args = parse_args()

    config = load_toml(args.config)
    cameras_json = config["data"]["cameras_json"]
    output_dir = config["data"]["output_dir"]
    image_dir = config["data"]["image_dir"]
    mesh_path = os.path.join(output_dir, "hair_mesh.ply")
    visualize_dir = os.path.join(output_dir, "visualize")

    print(f"Loading mesh: {mesh_path}")
    expected_vertices, expected_faces = read_ply_header_counts(mesh_path)
    tm = load_mesh(mesh_path)
    print(f"  Vertices: {len(tm.vertices):,}, Faces: {len(tm.faces):,}")
    if expected_vertices is not None and expected_faces is not None:
        print(f"  PLY header: {expected_vertices:,} vertices, {expected_faces:,} faces")
        if len(tm.vertices) != expected_vertices or len(tm.faces) != expected_faces:
            raise ValueError(
                "Loaded mesh size does not match PLY header "
                f"({len(tm.vertices)} vs {expected_vertices} vertices, "
                f"{len(tm.faces)} vs {expected_faces} faces)"
            )

    render_chunks = build_render_chunks(tm, args.chunk_vertices)
    del tm

    print(f"Loading cameras: {cameras_json}")
    cameras = load_cameras(cameras_json)
    print(f"  Found {len(cameras)} cameras")

    cam_ids = args.cam_ids if args.cam_ids else sorted(cameras.keys())

    os.makedirs(visualize_dir, exist_ok=True)

    first_img_path = image_dir % int(cam_ids[0])
    first_img = Image.open(first_img_path)
    width, height = first_img.size

    print(f"  Render resolution: {width} x {height}")

    renderer = pyrender.OffscreenRenderer(width, height)
    report_gl_backend(renderer)

    print(f"  Pre-building {len(render_chunks)} GPU mesh(es)...")
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[1.0, 1.0, 1.0])
    for chunk in render_chunks:
        normals = chunk["normals"]
        rgb = ((normals + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
        rgba = np.column_stack([rgb, np.full(len(rgb), 255, dtype=np.uint8)])
        chunk_tm = trimesh.Trimesh(
            vertices=chunk["vertices"], faces=chunk["faces"], process=False
        )
        chunk_tm.visual = trimesh.visual.ColorVisuals(vertex_colors=rgba)
        pm = pyrender.Mesh.from_trimesh(chunk_tm, smooth=True)
        for primitive in pm.primitives:
            if primitive.indices is not None:
                primitive.indices = np.asarray(primitive.indices, dtype=np.uint32)
            primitive.material.doubleSided = True
        scene.add(pm)

    camera = pyrender.IntrinsicsCamera(
        fx=1.0, fy=1.0, cx=0.5, cy=0.5, znear=1.0, zfar=100000.0
    )
    camera_node = scene.add(camera, pose=np.eye(4))

    cv_to_gl = np.diag([1.0, -1.0, -1.0])

    print(f"  Rendering {len(cam_ids)} views ({args.frame}-space normals)...")
    for i, cam_id in enumerate(cam_ids, 1):
        if cam_id not in cameras:
            print(f"  [{i}/{len(cam_ids)}] cam {cam_id}... SKIPPED")
            continue

        cam = cameras[cam_id]
        K = cam["K"]
        R = cam["E"][:3, :3]

        camera.fx = float(K[0, 0])
        camera.fy = float(K[1, 1])
        camera.cx = float(K[0, 2])
        camera.cy = float(K[1, 2])
        scene.set_pose(camera_node, extrinsic_to_gl_pose(cam["E"]))

        color, _ = renderer.render(
            scene, flags=pyrender.RenderFlags.FLAT | pyrender.RenderFlags.RGBA
        )

        if args.frame == "camera":
            # Rotate world-space normals into camera space.
            n_world = color[:, :, :3].astype(np.float32) / 127.5 - 1.0
            M = (cv_to_gl @ R).astype(np.float32)
            n_cam = n_world @ M.T
            rgb = ((n_cam + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        else:
            rgb = color[:, :, :3].copy()

        # Premultiply by alpha so partial-coverage silhouette pixels
        # composite cleanly onto a black background (no aliasing halo).
        alpha = color[:, :, 3:4].astype(np.float32) / 255.0
        rgb = (rgb.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)

        out_path = os.path.join(visualize_dir, f"{cam_id}.png")
        Image.fromarray(rgb).save(out_path)
        print(f"  [{i}/{len(cam_ids)}] cam {cam_id}... OK")

    renderer.delete()
    print(f"\nSaved {len(cam_ids)} normal maps to {visualize_dir}/")


if __name__ == "__main__":
    main()
