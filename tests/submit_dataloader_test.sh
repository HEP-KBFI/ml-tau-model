#!/bin/bash

# Test submission script to compare Raw (Parquet) and Preprocessed dataloaders.
# Usage: ./tests/submit_dataloader_test.sh

# Get the current git revision
TAG=$(date +%m%d)
GIT_REV=$(git rev-parse --short HEAD)
echo "Current Git Revision: $GIT_REV"

echo "Submitting MultiParTau with Raw (Parquet) dataloader..."
sbatch --job-name=raw_multipartau train-gpu.sh \
    training.model.name=MultiParTau \
    training.dataloader.type=parquet \
    training.dataloader.num_dataloader_workers=4 \
    training.dataloader.row_groups_per_read=16 \
    output_dir=outputs/${TAG}_multipartau_raw_${GIT_REV}

echo "Submitting MultiParTau with Preprocessed dataloader..."
sbatch --job-name=pre_multipartau train-gpu.sh \
    training.model.name=MultiParTau \
    training.dataloader.type=preprocessed \
    training.dataloader.num_dataloader_workers=0 \
    output_dir=outputs/${TAG}_multipartau_pre_${GIT_REV}

echo "All test jobs submitted. Monitor status with 'squeue -u $USER'"
