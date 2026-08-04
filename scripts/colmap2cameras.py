import argparse
import json
from pathlib import Path

import pycolmap

def parse_args():
    parser = argparse.ArgumentParser(description='Convert COLMAP camera params to cameras.json format.')
    parser.add_argument('-i', '--input_sparse_dir', type=Path, required=True, help='Path to the COLMAP sparse directory.')
    parser.add_argument('-o', '--output_json', type=Path, required=True, help='Path to the output cameras.json file.')
    return parser.parse_args()


def main():
    args = parse_args()
    recon = pycolmap.Reconstruction(args.input_sparse_dir)

    out = {"cameras": {}}
    for image_id in sorted(recon.images.keys()):
        image = recon.images[image_id]
        camera = image.camera
        intrinsic = camera.calibration_matrix()
        extrinsic = image.cam_from_world().matrix()
        out['cameras'][str(image_id)] = dict(
            intrinsic=intrinsic.tolist(),
            extrinsic=extrinsic.tolist(),
            distortion=[0] * 5
        )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(out, f, indent=2)


if __name__ == '__main__':
    main()
