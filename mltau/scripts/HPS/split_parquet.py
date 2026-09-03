"""Split large parquet files (z_test, qq_test) into chunks of a fixed size
and save them into per-sample subdirectories.

Usage
-----
    python mltau/scripts/HPS/split_parquet.py \
        --z_input   /path/to/z_test.parquet   \
        --qq_input  /path/to/qq_test.parquet  \
        --z_output  /path/to/Z/               \
        --qq_output /path/to/QQ/              \
        --chunk_size 1000
"""

import os
import argparse
import awkward as ak


def split_and_save(input_path: str, output_dir: str, chunk_size: int) -> None:
    """Load a parquet file, split it into fixed-size chunks and save each chunk.

    Parameters
    ----------
    input_path : str
        Absolute path to the input .parquet file.
    output_dir : str
        Directory where the chunk files will be written.
    chunk_size : int
        Number of entries per output chunk file.
    """
    os.makedirs(output_dir, exist_ok=True)

    data = ak.from_parquet(input_path)
    n_entries = len(data)

    base_name = os.path.splitext(os.path.basename(input_path))[0]

    n_chunks = (n_entries + chunk_size - 1) // chunk_size
    n_digits = len(str(n_chunks - 1))

    print(f"[{base_name}] {n_entries} entries → {n_chunks} chunks of ≤{chunk_size}")

    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, n_entries)
        chunk = data[start:end]

        chunk_name = f"{base_name}_{str(i).zfill(n_digits)}.parquet"
        out_path = os.path.join(output_dir, chunk_name)
        ak.to_parquet(chunk, out_path)
        print(f"  saved {out_path}  ({end - start} entries)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split z_test and qq_test parquet files into fixed-size chunks."
    )
    parser.add_argument(
        "--z_input",
        required=True,
        help="Path to z_test.parquet",
    )
    parser.add_argument(
        "--qq_input",
        required=True,
        help="Path to qq_test.parquet",
    )
    parser.add_argument(
        "--z_output",
        required=True,
        help="Output directory for Z chunks",
    )
    parser.add_argument(
        "--qq_output",
        required=True,
        help="Output directory for QQ chunks",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="Number of entries per chunk (default: 1000)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_and_save(args.z_input, args.z_output, args.chunk_size)
    split_and_save(args.qq_input, args.qq_output, args.chunk_size)


if __name__ == "__main__":
    main()
