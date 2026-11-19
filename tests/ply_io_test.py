"""Run with python tests/ply_io_test.py path/to/hair_recon on a CUDA machine."""

import argparse
from pathlib import Path
import struct
import subprocess
from tempfile import TemporaryDirectory


def check_round_trip(executable, directory):
    # Zero mean-shift iterations pass the two positions and directions through.
    config = directory / 'config.toml'
    config.write_text('[data]\noutput_dir = "."\n[meanshift]\n'
                      'min_neighbors = 0\nmax_iterations = 0\n')
    source = directory / 'fused.ply'
    source.write_text(
        'ply\nformat ascii 1.0\nelement vertex 2\n'
        'property float x\nproperty uchar red\nproperty float y\nproperty uchar green\n'
        'property float z\nproperty uchar blue\nproperty float normal_x\n'
        'property float normal_y\nproperty float normal_z\nend_header\n'
        '1 200 2 100 3 50 1 0 0\n4 11 5 22 6 33 0 1 0\n'
    )
    expected = (struct.pack('<6f3B', 1, 2, 3, 1, 0, 0, 59, 87, 177) +
                struct.pack('<6f3B', 4, 5, 6, 0, 1, 0, 190, 149, 177))

    output = directory / 'meanshift.ply'
    for input_format in ('ASCII', 'binary'):
        subprocess.run([executable, 'meanshift', config], cwd=directory, check=True)
        header, records = output.read_bytes().split(b'end_header\n', 1)
        lines = header.splitlines()
        assert b'format binary_little_endian 1.0' in lines
        assert b'element vertex 2' in lines
        for channel in (b'red', b'green', b'blue'):
            assert b'property uchar ' + channel in lines
        assert records == expected, f'Incorrect XYZ, normals or RGB for {input_format} input'
        output.replace(source)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('hair_recon', type=Path)
    executable = parser.parse_args().hair_recon.resolve()
    with TemporaryDirectory() as temp:
        check_round_trip(executable, Path(temp))
