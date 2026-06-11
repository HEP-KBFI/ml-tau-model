#!/usr/bin/env python3

import os
import argparse
from pathlib import Path
from huggingface_hub import HfApi

# Set timeout for large file uploads
os.environ["HF_HUB_ETAG_TIMEOUT"] = "1000"


def get_dir_size(path, ignore_patterns=None):
    """Calculate the total size of a directory in bytes."""
    from fnmatch import fnmatch

    total = 0
    for p in Path(path).rglob("*"):
        if p.is_file():
            if ignore_patterns:
                skip = False
                for pattern in ignore_patterns:
                    if fnmatch(p.name, pattern) or fnmatch(str(p.relative_to(path)), pattern):
                        skip = True
                        break
                if skip:
                    continue
            total += p.stat().st_size
    return total


def format_size(num, suffix="B"):
    """Convert bytes to human-readable format."""
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def main():
    parser = argparse.ArgumentParser(description="Upload ML-Tau model checkpoints to HuggingFace.")
    parser.add_argument("experiment_dir", help="Path to the experiment directory (output_dir)")
    parser.add_argument("--repo", default="HEP-KBFI/fcc-tau", help="HF repository ID")
    parser.add_argument("--name", help="Custom descriptive name for the experiment in the repo (defaults to directory name)")
    parser.add_argument("--path-prefix", default="", help="Prefix path in the repository (e.g. 'cld/0609_single')")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing")
    parser.add_argument("--repo-type", default="model", help="HF repository type (default: model)")

    args = parser.parse_args()

    exp_path = Path(args.experiment_dir)
    if not exp_path.is_dir():
        print(f"Error: Experiment directory {exp_path} not found.")
        return

    exp_name = args.name if args.name else exp_path.name
    
    if args.path_prefix:
        remote_path = f"{args.path_prefix.strip('/')}/{exp_name}"
    else:
        remote_path = f"{exp_name}"

    print(f"Preparing upload of {exp_path} to {args.repo}/{remote_path}...")

    # Find the best checkpoint
    models_dir = exp_path / "models"
    best_checkpoint = models_dir / "ParT-model_best.ckpt"

    if not best_checkpoint.exists():
        print(f"Warning: Best checkpoint {best_checkpoint} not found. Searching for any .ckpt in {models_dir}...")
        checkpoints = sorted(list(models_dir.glob("*.ckpt")), key=lambda x: x.stat().st_mtime)
        if checkpoints:
            best_checkpoint = checkpoints[-1]
            print(f"Using latest checkpoint: {best_checkpoint.name}")
        else:
            print(f"Error: No checkpoints found in {models_dir}")
            return

    api = HfApi() if not args.dry_run else None

    total_size = 0

    # Upload best weights
    if args.dry_run:
        print(f"[DRY-RUN] Would upload {best_checkpoint} as {remote_path}/models/{best_checkpoint.name}")
    else:
        print(f"Uploading {best_checkpoint} as {remote_path}/models/{best_checkpoint.name}...")
        api.upload_file(
            path_or_fileobj=str(best_checkpoint),
            path_in_repo=f"{remote_path}/models/{best_checkpoint.name}",
            repo_id=args.repo,
            repo_type=args.repo_type,
        )
    total_size += best_checkpoint.stat().st_size

    # Directories to upload
    # (local_name, remote_name, optional_ignore_patterns)
    dirs_to_upload = [
        ("logs", "logs"),
        ("tensorboard", "tensorboard"),
        ("predictions", "predictions"),
        (".hydra", "config"),
    ]

    # Upload folders
    for item in dirs_to_upload:
        local_dir_name = item[0]
        remote_dir_name = item[1]
        ignore_patterns = item[2] if len(item) > 2 else None

        local_dir_path = exp_path / local_dir_name
        if local_dir_path.exists():
            size = get_dir_size(local_dir_path, ignore_patterns=ignore_patterns)
            total_size += size
            if args.dry_run:
                print(f"[DRY-RUN] Would upload folder {local_dir_path} ({format_size(size)}) to {remote_path}/{remote_dir_name}")
            else:
                print(f"Uploading folder {local_dir_path} ({format_size(size)}) to {remote_path}/{remote_dir_name}...")
                api.upload_folder(
                    folder_path=str(local_dir_path),
                    path_in_repo=f"{remote_path}/{remote_dir_name}",
                    repo_id=args.repo,
                    repo_type=args.repo_type,
                    ignore_patterns=ignore_patterns,
                )

    # Upload all PDF files in the experiment directory recursively
    # This captures validation plots
    for pdf_file in exp_path.rglob("*.pdf"):
        # Relative path from exp_path to preserve structure if they are in subdirs
        rel_path = pdf_file.relative_to(exp_path)
        if args.dry_run:
            print(f"[DRY-RUN] Would upload {pdf_file} as {remote_path}/{rel_path}")
        else:
            print(f"Uploading {pdf_file} as {remote_path}/{rel_path}...")
            api.upload_file(
                path_or_fileobj=str(pdf_file),
                path_in_repo=f"{remote_path}/{rel_path}",
                repo_id=args.repo,
                repo_type=args.repo_type,
            )
        total_size += pdf_file.stat().st_size

    print(f"Total size processed: {format_size(total_size)}")


if __name__ == "__main__":
    main()
