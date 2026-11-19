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


def check_strand_export(executable, directory):
    config = directory / 'clean.toml'
    config.write_text('[data]\noutput_dir = "."\n[clean]\n'
                      'min_length = 0.0\noutlier_radius = 0.0\n')
    first = (1, 2, 3, 2, 0, 0)
    second = (4, 5, 6, -1, 0, 0)
    third = (7, 8, 9, 0, 0.5, 0)
    # The empty middle strand leaves a gap between the exported strand indices.
    strands = bytearray(struct.pack('<i', 3))
    for points in ((first, second), (), (third,)):
        strands.extend(struct.pack('<i', len(points)))
        for point in points:
            strands.extend(struct.pack('<6f', *point))
    vertex = struct.Struct('<6fI3B')
    expected = (vertex.pack(*first, 0, 59, 87, 177) +
                vertex.pack(*second, 0, 59, 87, 177) +
                vertex.pack(*third, 2, 190, 149, 177))
    axes = ('x', 'y', 'z', 'nx', 'ny', 'nz')
    properties = ([f'property float {axis}' for axis in axes] +
                  ['property uint label'] +
                  [f'property uchar {channel}' for channel in ('red', 'green', 'blue')])

    output = directory / 'strands_clean.ply'
    for source, expected_records in ((strands, expected), (struct.pack('<i', 0), b'')):
        (directory / 'strands_raw.bin').write_bytes(source)
        subprocess.run([executable, 'clean', config], cwd=directory, check=True)
        header, records = output.read_bytes().split(b'end_header\n', 1)
        lines = header.decode('ascii').splitlines()
        assert 'format binary_little_endian 1.0' in lines
        assert f'element vertex {len(expected_records) // vertex.size}' in lines
        assert [line for line in lines if line.startswith('property ')] == properties
        assert records == expected_records
        assert (directory / 'strands_clean.bin').read_bytes() == source


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('hair_recon', type=Path)
    executable = parser.parse_args().hair_recon.resolve()
    with TemporaryDirectory() as temp:
        check_round_trip(executable, Path(temp))
        check_strand_export(executable, Path(temp))
