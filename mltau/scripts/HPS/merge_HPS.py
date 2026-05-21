"""Merge per-chunk HPS parquet files into a single file per sample.

Usage:
    python merge_HPS.py --input_dir /home/laurits/HPS

Expects:
    <input_dir>/Z/   — chunk parquet files for the Z→ττ sample
    <input_dir>/QQ/  — chunk parquet files for the Z→qq sample

Produces:
    <input_dir>/hps_z.parquet
    <input_dir>/hps_qq.parquet

For QQ a two-pass strategy is used: files are first merged in batches of
QQ_CHUNK_SIZE, then the resulting in-memory arrays are concatenated together.
This avoids holding all QQ files in memory simultaneously.
"""

import os
import glob
import argparse
import awkward as ak

QQ_CHUNK_SIZE = 100


def merge_sample(sample_dir: str, output_path: str) -> None:
    """Load all files at once and write a single parquet (suitable for Z)."""
    pattern = os.path.join(sample_dir, "*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {sample_dir}")
    print(f"  Merging {len(files)} files from {sample_dir} → {output_path}")
    merged = ak.concatenate([ak.from_parquet(f) for f in files])
    ak.to_parquet(merged, output_path)
    print(f"  Done. Written {len(merged)} events.")


def merge_sample_chunked(
    sample_dir: str, output_path: str, chunk_size: int = QQ_CHUNK_SIZE
) -> None:
    """Two-pass disk-based merge to keep peak memory low.

    Pass 1: merge ``chunk_size`` input files at a time and write each result
            to a temporary parquet file on disk.
    Pass 2: concatenate the (much smaller set of) temp files into the final
            output, then delete the temp files.

    Peak memory is proportional to one chunk of ``chunk_size`` files.
    """
    pattern = os.path.join(sample_dir, "*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {sample_dir}")

    n_chunks = (len(files) + chunk_size - 1) // chunk_size
    tmp_dir = os.path.join(sample_dir, "_tmp_merge")
    os.makedirs(tmp_dir, exist_ok=True)
    print(
        f"  Merging {len(files)} files from {sample_dir} in {n_chunks} chunk(s) of "
        f"{chunk_size} → {output_path}"
    )

    # Pass 1: write one temp file per chunk
    tmp_files = []
    for i in range(0, len(files), chunk_size):
        batch = files[i : i + chunk_size]
        chunk_idx = i // chunk_size + 1
        print(f"    pass 1 — chunk {chunk_idx}/{n_chunks}: {len(batch)} files")
        merged_chunk = ak.concatenate([ak.from_parquet(f) for f in batch])
        tmp_path = os.path.join(tmp_dir, f"chunk_{chunk_idx:04d}.parquet")
        ak.to_parquet(merged_chunk, tmp_path)
        tmp_files.append(tmp_path)
        del merged_chunk  # free memory immediately

    # Pass 2: concatenate temp files into the final output
    print(f"    pass 2 — merging {len(tmp_files)} chunk files into {output_path}")
    merged = ak.concatenate([ak.from_parquet(f) for f in tmp_files])
    ak.to_parquet(merged, output_path)
    n_events = len(merged)
    del merged

    # Clean up temp files
    for f in tmp_files:
        os.remove(f)
    os.rmdir(tmp_dir)

    print(f"  Done. Written {n_events} events.")


def main():
    parser = argparse.ArgumentParser(
        description="Merge HPS chunk parquet files into one file per sample."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing Z/ and QQ/ subdirectories with chunk parquet files.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir

    print("\n[Z]")
    merge_sample(
        sample_dir=os.path.join(input_dir, "Z"),
        output_path=os.path.join(input_dir, "hps_z.parquet"),
    )

    print("\n[QQ]")
    merge_sample_chunked(
        sample_dir=os.path.join(input_dir, "QQ"),
        output_path=os.path.join(input_dir, "hps_qq.parquet"),
    )


if __name__ == "__main__":
    main()
