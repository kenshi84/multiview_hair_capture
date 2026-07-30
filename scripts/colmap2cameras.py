import argparse
import json
from pathlib import Path

import pycolmap

def parse_args():
    parser = argparse.ArgumentParser(description='Convert COLMAP camera params to cameras.json format.')
    parser.add_argument('-c', '--case_dir', type=Path, required=True, help='Path to the case directory.')
    parser.add_argument('-s', '--scene_scale', type=float, required=True, help='Scale factor for the scene.')
    return parser.parse_args()


def main():
    args = parse_args()
    recon = pycolmap.Reconstruction(args.case_dir / 'sparse')

    out = {"cameras": {}}
    for image_id in sorted(recon.images.keys()):
        image = recon.images[image_id]
        camera = image.camera
        intrinsic = camera.calibration_matrix()
        extrinsic = image.cam_from_world().matrix()
        extrinsic[:3, 3] *= args.scene_scale  # Scale translation
        out['cameras'][str(image_id)] = dict(
            intrinsic=intrinsic.tolist(),
            extrinsic=extrinsic.tolist(),
            distortion=[0] * 5
        )
    with open(args.case_dir / 'cameras.json', 'w') as f:
        json.dump(out, f, indent=2)
    return len(out['cameras'])


if __name__ == '__main__':
    main()
