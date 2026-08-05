#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Download and convert Multiface data for the hair capture pipeline.

Downloads one frame from all cameras for a given subject/expression,
then converts the data into the format expected by the pipeline
(cameras.json, flat images, TOML config).

Skips downloading if the data already exists on disk.

Usage:
    # Default subject (002421669), all cameras:
    python scripts/prepare_multiface.py

    # Specific subject, limit cameras for quick test:
    python scripts/prepare_multiface.py --subject_id 5067077 --num_cameras 1

    # Skip download, just re-run conversion on existing data:
    python scripts/prepare_multiface.py --subject_id 5067077
"""

import argparse
import glob
import io
import json
import math
import os
import shutil
import statistics
import sys
import tarfile
import time

try:
    import requests
except ImportError:
    requests = None

S3_BASE = (
    "https://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15.s3.amazonaws.com"
    "/MugsyDataRelease/v0.0/identities"
)

SUBJECT_V1_FOLDERS = {
    "002757580": "m--20171024--0000--002757580--GHS",
    "002539136": "m--20180105--0000--002539136--GHS",
    "6674443": "m--20180226--0000--6674443--GHS",
    "6795937": "m--20180227--0000--6795937--GHS",
    "8870559": "m--20180406--0000--8870559--GHS",
    "2183941": "m--20180418--0000--2183941--GHS",
    "002643814": "m--20180426--0000--002643814--GHS",
    "5372021": "m--20180510--0000--5372021--GHS",
    "7889059": "m--20180927--0000--7889059--GHS",
    "002914589": "m--20181017--0000--002914589--GHS",
}
SUBJECT_V2_FOLDERS = {
    "002421669": "m--20190529--1300--002421669--GHS",
    "5067077": "m--20190529--1004--5067077--GHS",
    "002645310": "m--20190828--1318--002645310--GHS",
}

MULTIFACE_REPO = os.path.join(os.path.dirname(__file__), "..", "..", "multiface")


def parse_args():
    p = argparse.ArgumentParser(
        description="Download and convert Multiface data for hair capture pipeline"
    )
    p.add_argument("--subject_id", default="002421669")
    p.add_argument("--expression", default="EXP_eye_neutral", help="V1 defaults to E001_Neutral_Eyes_Open")
    p.add_argument("--frame_index", default=None,
                    help="6-digit frame index; auto-detect if omitted")
    p.add_argument("--download_dir", default="data/multiface",
                    help="Raw download directory")
    p.add_argument("--output_dir", default="data",
                    help="Pipeline data directory (cameras.json + images go under <output_dir>/<subject_folder>/)")
    p.add_argument("--num_cameras", type=int, default=None,
                    help="Limit number of cameras (for quick testing)")
    p.add_argument("--max_retries", type=int, default=3)
    p.add_argument("--proxy", default=None,
                    help="HTTP proxy URL (auto-detected from HTTP_PROXY env var)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def make_session(proxy=None):
    session = requests.Session()
    if proxy:
        session.proxies = {"http": proxy, "https": proxy}
    elif os.environ.get("HTTP_PROXY"):
        p = os.environ["HTTP_PROXY"]
        session.proxies = {"http": p, "https": p}
    elif os.environ.get("HTTPS_PROXY"):
        p = os.environ["HTTPS_PROXY"]
        session.proxies = {"http": p, "https": p}
    return session


def download_and_extract_metadata(subject_id, dest_dir, session, max_retries=3):
    subject_folder = SUBJECT_V2_FOLDERS[subject_id] if subject_id in SUBJECT_V2_FOLDERS else SUBJECT_V1_FOLDERS[subject_id]
    metadata_dir = os.path.join(dest_dir, subject_folder)

    krt_path = os.path.join(metadata_dir, "KRT")
    frame_list_path = os.path.join(metadata_dir, "frame_list.txt")
    if os.path.exists(krt_path) and os.path.exists(frame_list_path):
        print("  Already exists, skipping.")
        return krt_path, frame_list_path

    url = f"{S3_BASE}/{subject_id}/metadata.tar"
    os.makedirs(dest_dir, exist_ok=True)
    print(f"  Downloading {url}")

    for attempt in range(max_retries):
        try:
            resp = session.get(url, timeout=120)
            resp.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                print(f"  Retry {attempt + 1}/{max_retries}: {e}")
                time.sleep(2 ** attempt)
            else:
                raise

    with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r") as tar:
        tar.extractall(path=dest_dir, filter="data")

    if not os.path.exists(krt_path):
        raise FileNotFoundError(f"KRT not found at {krt_path}")
    if not os.path.exists(frame_list_path):
        raise FileNotFoundError(f"frame_list.txt not found at {frame_list_path}")

    return krt_path, frame_list_path


def stream_extract_single_frame(url, target_frame, output_path, session,
                                max_retries=3):
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return True

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    target_suffix = f"/{target_frame}.png"

    for attempt in range(max_retries):
        try:
            resp = session.get(url, stream=True, timeout=300)
            resp.raise_for_status()
            try:
                with tarfile.open(fileobj=resp.raw, mode="r|") as tar:
                    for member in tar:  # noqa: PERF203
                        # Manual per-member extract; equivalent of filter="data".
                        if member.islnk() or member.issym():
                            continue
                        if member.name.endswith(target_suffix):
                            fobj = tar.extractfile(member)
                            if fobj is None:
                                continue
                            with open(output_path, "wb") as out:
                                out.write(fobj.read())
                            return True
            except Exception:
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    return True
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise
            finally:
                resp.close()
            return False
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                print(f"\n  Retry {attempt + 1}/{max_retries}: {e}")
                time.sleep(2 ** attempt)
            else:
                raise
    return False


def stream_extract_all_frames(url, camera_ids, target_frame, output_exp_dir, session,
                              max_retries=3):
    output_paths = [os.path.join(output_exp_dir, cam_id, f"{target_frame}.png") for cam_id in camera_ids]
    if all(os.path.exists(p) and os.path.getsize(p) > 0 for p in output_paths):
        return []
    failed = []
    target_suffix = f"/{target_frame}.png"
    for attempt in range(max_retries):
        try:
            resp = session.get(url, stream=True, timeout=300)
            resp.raise_for_status()
            try:
                with tarfile.open(fileobj=resp.raw, mode="r|") as tar:
                    for member in tar:  # noqa: PERF203
                        if member.islnk() or member.issym():
                            continue
                        if member.name.endswith(target_suffix):
                            cam_id = os.path.basename(os.path.dirname(member.name))
                            if cam_id not in camera_ids:
                                continue
                            fobj = tar.extractfile(member)
                            if fobj is None:
                                failed.append(cam_id)
                                continue
                            output_path = os.path.join(output_exp_dir, cam_id, f"{target_frame}.png")
                            os.makedirs(os.path.dirname(output_path), exist_ok=True)
                            with open(output_path, "wb") as out:
                                out.write(fobj.read())
                            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                                failed.append(cam_id)
            except Exception:
                if all(os.path.exists(output_path) and os.path.getsize(output_path) > 0 for output_path in output_paths):
                    return []
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise
            finally:
                resp.close()
            return failed
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                print(f"\n  Retry {attempt + 1}/{max_retries}: {e}")
                time.sleep(2 ** attempt)
            else:
                raise
    return failed

def download_images(subject_id, subject_folder, expression, target_frame,
                    camera_ids, dest_dir, session, version, max_retries=3):
    failed = []
    t0 = time.time()

    if version == "V1":
        output_exp_dir = os.path.join(
            dest_dir, subject_folder,
            "images", expression)
        url = f"{S3_BASE}/{subject_id}/images--{expression}.tar"
        print(f"  Downloading {url}...")
        try:
            failed = stream_extract_all_frames(url, camera_ids, target_frame, output_exp_dir, session, max_retries)
        except Exception as e:
            print(f"FAILED ({e})")
            failed.extend(camera_ids)
    elif version == "V2":
        for i, cam_id in enumerate(camera_ids, start=1):
            output_path = os.path.join(
                dest_dir, subject_folder,
                "images", expression, cam_id, f"{target_frame}.png",
            )
            url = f"{S3_BASE}/{subject_id}/images--{expression}_cam{cam_id}.tar"

            print(f"  [{i}/{len(camera_ids)}] cam {cam_id}...", end=" ", flush=True)
            try:
                ok = stream_extract_single_frame(
                    url, target_frame, output_path, session, max_retries
                )
                if ok:
                    print("OK")
                else:
                    print("FRAME NOT FOUND")
                    failed.append(cam_id)
            except Exception as e:
                print(f"FAILED ({e})")
                failed.append(cam_id)
    else:
        raise ValueError(f"Unknown version: {version}")

    elapsed = time.time() - t0
    print(f"  {len(camera_ids) - len(failed)}/{len(camera_ids)} cameras "
          f"in {elapsed:.0f}s")
    return failed


# ---------------------------------------------------------------------------
# Frame list / camera list helpers
# ---------------------------------------------------------------------------

def parse_frame_list(frame_list_path, expression):
    frames = []
    with open(frame_list_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2 and parts[0] == expression:
                frames.append(parts[1])

    if not frames:
        all_expressions = set()
        with open(frame_list_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 1:
                    all_expressions.add(parts[0])
        raise ValueError(
            f"Expression '{expression}' not found.\n"
            f"Available: {sorted(all_expressions)}"
        )

    frames.sort()
    return frames


def get_camera_ids(subject_id):
    config_path = os.path.join(
        MULTIFACE_REPO, "camera_configs",
        f"camera-split-config_{subject_id}.json",
    )
    config_path = os.path.normpath(config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Camera config not found: {config_path}\n"
            f"Make sure the multiface repo is at {MULTIFACE_REPO}"
        )
    with open(config_path, "r") as f:
        config = json.load(f)
    return sorted(set(config["full"]["train"]))


def detect_frame_index(download_dir, subject_folder, expression, camera_ids):
    for cam_id in camera_ids:
        img_dir = os.path.join(
            download_dir, subject_folder, "images", expression, cam_id
        )
        if os.path.isdir(img_dir):
            pngs = sorted(glob.glob(os.path.join(img_dir, "*.png")))
            if pngs:
                return os.path.splitext(os.path.basename(pngs[0]))[0]
    return None


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def parse_krt(krt_path):
    cameras = {}
    with open(krt_path, "r") as f:
        while True:
            name = f.readline()
            if name == "":
                break
            name = name.strip()
            if not name:
                continue

            intrinsic = []
            for _ in range(3):
                row = [float(x) for x in f.readline().split()]
                intrinsic.append(row)

            dist = [float(x) for x in f.readline().split()]

            extrinsic = []
            for _ in range(3):
                row = [float(x) for x in f.readline().split()]
                extrinsic.append(row)

            f.readline()

            cameras[name] = {
                "intrinsic": intrinsic,
                "extrinsic": extrinsic,
                "distortion": (dist + [0.0] * 5)[:5],
            }
    return cameras


def _mat3x3_transpose(R):
    return [[R[j][i] for j in range(3)] for i in range(3)]


def _mat3x3_vec3_mul(M, v):
    return [sum(M[r][c] * v[c] for c in range(3)) for r in range(3)]


def _vec3_norm(v):
    return math.sqrt(sum(x * x for x in v))


def compute_camera_centers(cameras):
    centers = {}
    for cam_id, cam in cameras.items():
        E = cam["extrinsic"]
        R = [E[r][:3] for r in range(3)]
        t = [E[r][3] for r in range(3)]
        Rt = _mat3x3_transpose(R)
        center = _mat3x3_vec3_mul(Rt, [-x for x in t])
        centers[cam_id] = center
    return centers


def check_and_fix_units(cameras):
    centers = compute_camera_centers(cameras)
    distances = [_vec3_norm(c) for c in centers.values()]
    median_dist = statistics.median(distances)
    print(f"  Median camera distance from origin: {median_dist:.2f}")

    if median_dist < 10:
        print("  Detected meters, converting to millimeters...")
        for cam in cameras.values():
            for r in range(3):
                cam["extrinsic"][r][3] *= 1000.0
    elif median_dist > 100:
        print("  Units: mm")
    else:
        print("  WARNING: Ambiguous unit scale. Assuming millimeters.")


def estimate_depth_range(cameras, margin=0.3):
    centers = list(compute_camera_centers(cameras).values())
    n = len(centers)
    centroid = [sum(c[i] for c in centers) / n for i in range(3)]
    distances = [_vec3_norm([c[i] - centroid[i] for i in range(3)]) for c in centers]
    avg_dist = statistics.mean(distances)
    min_depth = max(10.0, round(avg_dist * (1 - margin) / 10) * 10)
    max_depth = round(avg_dist * (1 + margin) / 10) * 10
    return float(min_depth), float(max_depth)


def write_cameras_json(cameras, output_path):
    out = {"cameras": {}}
    for cam_id in sorted(cameras.keys()):
        cam = cameras[cam_id]
        out["cameras"][cam_id] = {
            "intrinsic": cam["intrinsic"],
            "extrinsic": cam["extrinsic"],
            "distortion": cam["distortion"],
        }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    return len(out["cameras"])


def copy_and_rename_images(download_dir, output_dir, subject_id,
                           subject_folder, expression, frame_index,
                           camera_ids):
    images_dir = os.path.join(output_dir, subject_id, "images")
    os.makedirs(images_dir, exist_ok=True)
    copied, missing = [], []
    for cam_id in camera_ids:
        src = os.path.join(
            download_dir, subject_folder,
            "images", expression, cam_id, f"{frame_index}.png",
        )
        dst = os.path.join(images_dir, f"{cam_id}.png")
        if not os.path.exists(src):
            missing.append(cam_id)
            continue
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            copied.append(cam_id)
            continue
        shutil.copy2(src, dst)
        copied.append(cam_id)
    return copied, missing


def generate_toml_config(output_path, subject_id, data_dir,
                         min_depth, max_depth,
                         distorted_images=False):
    distorted_str = "true" if distorted_images else "false"
    base = f"{data_dir}/{subject_id}".replace("\\", "/")
    out_dir = f"output/{subject_id}".replace("\\", "/")

    content = f"""\
# Multiface subject configuration
# Generated by scripts/prepare_multiface.py

[data]
cameras_json = "{base}/cameras.json"
image_dir = "{base}/images/%06d.png"
mask_dir = "{base}/hair_masks/%06d.png"
output_dir = "{out_dir}"
downsample = 1.0
distorted_images = {distorted_str}

[mvs]
min_depth = {min_depth:.0f}
max_depth = {max_depth:.0f}
num_hierarchy_levels = 4
gaussian_pyramid = false
num_neighbor_views = 25
min_angle = 1.0
max_angle = 90.0
num_view_select = 4
num_gpus = 1
alpha = 0.1
pt_sample_radius = 10.0
pt_sample_kappa = 41
use_mask = false
mask_min_neighbor_views = 1
iterations = 8
patch_size = 21
delta_depth = 30.0
delta_orient = 0.8
spatial_prop_radius = 5.0

[mvs.gabor]
num_orientations = 180
kernel_size = 21
sigma = 1.12
gamma = 0.28
lambda = 3.00
min_contrast = 0.01
min_response = 0.02

[mvs.fusion]
position_threshold = 1.0
orientation_threshold = 10.0
min_neighbors = 4
cost_threshold = 0.2

[meanshift]
neighbor_radius = 2.0
min_neighbors = 10
sigma_position = 0.1
sigma_orientation = 30.0
convergence = 0.002
max_iterations = 1000

[trace]
step_size = 0.1
neighbor_radius = 0.1
angle_threshold = 30.0
min_strand_length = 2.0

[clean]
min_length = 5.0
outlier_radius = 10.0
outlier_min_neighbors = 3

[debug]
gpu_id = 0
log_level = "info"
log_file = ""
save_intermediates = false
profile = true
"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(content)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    subject_folder = SUBJECT_V2_FOLDERS.get(args.subject_id)
    if not subject_folder:
        subject_folder = SUBJECT_V1_FOLDERS.get(args.subject_id)
        if not subject_folder:
            print(f"ERROR: Unknown subject_id '{args.subject_id}'")
            print(f"Available: {list(SUBJECT_V2_FOLDERS.keys()) + list(SUBJECT_V1_FOLDERS.keys())}")
            sys.exit(1)
    version = "V2" if args.subject_id in SUBJECT_V2_FOLDERS else "V1"

    if version == "V1" and args.expression == "EXP_eye_neutral":
        args.expression = "E001_Neutral_Eyes_Open"
        print(f"  Using expression '{args.expression}' for V1 subject {args.subject_id}")

    camera_ids = get_camera_ids(args.subject_id)
    if args.num_cameras:
        camera_ids = camera_ids[: args.num_cameras]

    # ------------------------------------------------------------------
    # Phase 1: Download (skipped if data already exists)
    # ------------------------------------------------------------------
    krt_path = os.path.join(args.download_dir, subject_folder, "KRT")
    frame_list_path = os.path.join(args.download_dir, subject_folder, "frame_list.txt")
    need_download = not (os.path.exists(krt_path) and os.path.exists(frame_list_path))

    if need_download and requests is None:
        print("ERROR: 'requests' package required for downloading. pip install requests")
        sys.exit(1)

    if need_download:
        session = make_session(args.proxy)
        print("Step 1: Downloading metadata...")
        krt_path, frame_list_path = download_and_extract_metadata(
            args.subject_id, args.download_dir, session, args.max_retries
        )
    else:
        print("Step 1: Metadata already exists, skipping download.")

    print(f"  KRT: {krt_path}")

    # Determine target frame
    frames = parse_frame_list(frame_list_path, args.expression)
    if args.frame_index:
        if args.frame_index not in frames:
            print(f"ERROR: Frame {args.frame_index} not available for {args.expression}")
            sys.exit(1)
        target_frame = args.frame_index
    else:
        target_frame = frames[0]
    print(f"  Expression: {args.expression}, frame: {target_frame}")

    # Check how many camera images already exist
    existing = 0
    for cam_id in camera_ids:
        img_path = os.path.join(
            args.download_dir, subject_folder,
            "images", args.expression, cam_id, f"{target_frame}.png",
        )
        if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
            existing += 1

    if existing < len(camera_ids):
        if requests is None:
            print(f"WARNING: Only {existing}/{len(camera_ids)} images exist "
                  f"and 'requests' not available for downloading.")
        else:
            if not need_download:
                session = make_session(args.proxy)
            print(f"\nStep 2: Downloading images ({existing}/{len(camera_ids)} "
                  f"already exist)...")
            failed = download_images(
                args.subject_id, subject_folder, args.expression, target_frame,
                camera_ids, args.download_dir, session, version, args.max_retries,
            )
            if failed:
                print(f"  Failed cameras: {failed}")
    else:
        print(f"\nStep 2: All {len(camera_ids)} images already exist, skipping download.")

    # ------------------------------------------------------------------
    # Phase 2: Convert
    # ------------------------------------------------------------------
    print(f"\nStep 3: Parsing KRT...")
    cameras = parse_krt(krt_path)
    print(f"  Found {len(cameras)} cameras")

    print("\nStep 4: Checking units...")
    check_and_fix_units(cameras)

    # Write cameras.json
    subject_data_dir = os.path.join(args.output_dir, args.subject_id)
    cameras_json_path = os.path.join(subject_data_dir, "cameras.json")
    print(f"\nStep 5: Writing {cameras_json_path}...")
    write_cameras_json(cameras, cameras_json_path)

    # Copy and rename images
    print(f"\nStep 6: Copying images...")
    copied, missing = copy_and_rename_images(
        args.download_dir, args.output_dir, args.subject_id,
        subject_folder, args.expression, target_frame,
        sorted(cameras.keys()),
    )
    print(f"  Copied: {len(copied)}, missing: {len(missing)}")

    if missing:
        for cam_id in missing:
            cameras.pop(cam_id, None)
        write_cameras_json(cameras, cameras_json_path)

    # Depth range
    min_depth, max_depth = estimate_depth_range(cameras)
    print(f"  Depth range: {min_depth:.0f} - {max_depth:.0f} mm")

    # Distortion
    all_zero_dist = all(
        all(abs(d) < 1e-10 for d in cam["distortion"])
        for cam in cameras.values()
    )

    # Generate config
    config_path = os.path.join("configs", f"{args.subject_id}.toml")
    print(f"\nStep 7: Generating {config_path}...")
    generate_toml_config(
        output_path=config_path,
        subject_id=args.subject_id,
        data_dir=args.output_dir,
        min_depth=min_depth,
        max_depth=max_depth,
        distorted_images=not all_zero_dist,
    )

    # Summary
    print(f"\n{'=' * 50}")
    print(f"  Cameras:      {len(cameras)}")
    print(f"  cameras.json: {cameras_json_path}")
    print(f"  Images:       {subject_data_dir}/images/")
    print(f"  Config:       {config_path}")
    print(f"  Depth range:  {min_depth:.0f} - {max_depth:.0f} mm")
    print(f"\nTo run the pipeline:")
    print(f"  ./build/hair_recon pipeline {config_path}")


if __name__ == "__main__":
    main()
