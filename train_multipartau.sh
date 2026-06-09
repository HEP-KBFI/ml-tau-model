#!/bin/bash

# This script trains the MultiParTau model on the datasets in 0412_increased_stats.
# It uses the project's hydra-based configuration system with appropriate overrides.

# Exit on error
set -e

# Default configuration
DATA_DIR="0412_increased_stats"
MODEL="MultiParTau"
OUTPUT_DIR="outputs/multipartau_standalone1"
BATCH_SIZE=12288

echo "--------------------------------------------------"
echo " Training MultiParTau model"
echo " Dataset: $DATA_DIR"
echo " Output:  $OUTPUT_DIR"
echo "--------------------------------------------------"

# Ensure the output directory exists
mkdir -p "$OUTPUT_DIR"

# We use the existing run.sh wrapper if available, as it handles the 
# apptainer container and PYTHONPATH correctly for this project's environment.
./run.sh python3 mltau/scripts/train.py \
    dataset.data_dir="$DATA_DIR" \
    training.model.name="$MODEL" \
    training.dataloader.batch_size=$BATCH_SIZE \
    output_dir="$OUTPUT_DIR" \
    "$@"
