import argparse
from pathlib import Path
import struct



HAIR_HEADER = struct.Struct("<4sIIII5f88s")
UINT16_MAX = (1 << 16) - 1


def read_exact(file, size):
    data = file.read(size)
    if len(data) != size:
        raise ValueError("Unexpected end of input file")
    return data


def scan_strands(input_file):
    """Return the segment count of each strand and the total point count."""
    with input_file.open("rb") as file:
        strand_count = struct.unpack("<i", read_exact(file, 4))[0]
        if strand_count < 0:
            raise ValueError("Input file has a negative strand count")

        segments = bytearray(strand_count * 2)
        point_count = 0
        for strand_index in range(strand_count):
            points_in_strand = struct.unpack("<i", read_exact(file, 4))[0]
            if not 1 <= points_in_strand <= UINT16_MAX + 1:
                raise ValueError(
                    f"Strand {strand_index} has {points_in_strand} points; "
                    "HAIR supports 1 through 65536 points per strand"
                )
            struct.pack_into("<H", segments, strand_index * 2, points_in_strand - 1)
            point_count += points_in_strand
            file.seek(points_in_strand * 6 * 4, 1)

        if file.tell() != input_file.stat().st_size:
            raise ValueError("Input file is truncated or has trailing data")

    return strand_count, point_count, segments


def write_positions(file, records):
    """Write xyz values from x,y,z,dx,dy,dz float records."""
    positions = bytearray(len(records) // 2)
    source = memoryview(records)
    destination = memoryview(positions)
    for source_offset, destination_offset in zip(
        range(0, len(records), 24), range(0, len(positions), 12)
    ):
        destination[destination_offset : destination_offset + 12] = source[
            source_offset : source_offset + 12
        ]
    file.write(positions)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert .bin files to Cem Yuksel's .hair format.")
    parser.add_argument("-i", "--input_file", type=Path, required=True, help="Path to the input binary file.")
    parser.add_argument("-o", "--overwrite", action="store_true", help="Overwrite the output file if it exists.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_file = args.input_file.with_suffix(".hair")
    if output_file.exists() and not args.overwrite:
        print(f"Error: Output file {output_file} already exists. Use --overwrite to overwrite it.")
        return
    print(f"Converting {args.input_file} to {output_file}")         # Specified in https://www.cemyuksel.com/research/hairmodels/
    try:
        strand_count, point_count, segments = scan_strands(args.input_file)
        with args.input_file.open("rb") as input_file, output_file.open("wb") as file:
            read_exact(input_file, 4)  # strand count, already scanned above
            file.write(
                HAIR_HEADER.pack(
                    b"HAIR",
                    strand_count,
                    point_count,
                    0b11,  # segments and points arrays are present
                    0,
                    1.0,  # default thickness
                    0.0,  # default transparency
                    1.0,
                    1.0,
                    1.0,  # default color
                    b"",
                )
            )
            file.write(segments)

            for _ in range(strand_count):
                points_in_strand = struct.unpack("<i", read_exact(input_file, 4))[0]
                # The input stores xyz followed by a direction vector per point.
                write_positions(file, read_exact(input_file, points_in_strand * 6 * 4))
    except (OSError, ValueError) as error:
        print(f"Error: {error}")
        return


if __name__ == "__main__":
    main()
