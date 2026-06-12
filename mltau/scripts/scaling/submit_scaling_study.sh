#!/bin/bash

# Scaling study submission script for MultiParTau
# Dimensions: Dataset size and Model size

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Project root is two levels up
REPO_ROOT="$( cd "${SCRIPT_DIR}/../../../" && pwd )"

cd "${REPO_ROOT}"

if [[ -n $(git status --porcelain) ]]; then
  echo "Error: Git status is not clean. Please commit or stash your changes before submitting jobs."
  exit 1
fi

TAG="scaling_$(date +%m%d)"
GIT_REV=$(git rev-parse --short HEAD)
echo "Starting scaling study. Tag: ${TAG}, Git Rev: ${GIT_REV}"

# Define scaling dimensions
DATASET_SIZES=(100000 500000 1000000 "null")
MODEL_SIZES=("tiny" "small" "medium" "large")

for ds_size in "${DATASET_SIZES[@]}"; do
    for m_size in "${MODEL_SIZES[@]}"; do
        
        JOB_NAME="mp_ds${ds_size}_m${m_size}"
        OUTPUT_DIR="outputs/${TAG}/${JOB_NAME}_${GIT_REV}"
        
        # Base command
        CMD="sbatch --job-name=${JOB_NAME} train-gpu.sh \
            training.model.name=MultiParTau \
            dataset.limit_samples=${ds_size} \
            training.dataloader.batch_size=6144 \
            output_dir=${OUTPUT_DIR}"

        # Model-specific overrides
        if [ "$m_size" == "tiny" ]; then
            CMD="${CMD} training.model.num_layers=1 training.model.num_heads=2 training.model.num_cls_layers=1 \
                 training.model.embed_dims='[64,128,64]' training.model.pair_embed_dims='[16,16,16]'"
        elif [ "$m_size" == "small" ]; then
            CMD="${CMD} training.model.num_layers=1 training.model.num_heads=4 training.model.num_cls_layers=1 \
                 training.model.embed_dims='[128,256,128]' training.model.pair_embed_dims='[32,32,32]'"
        elif [ "$m_size" == "medium" ]; then
            # Medium is the previous default: 2 layers, 8 heads, [256, 512, 256]
            CMD="${CMD} training.model.num_layers=2 training.model.num_heads=8 training.model.num_cls_layers=2 \
                 training.model.embed_dims='[256,512,256]' training.model.pair_embed_dims='[64,64,64]'"
        elif [ "$m_size" == "large" ]; then
            CMD="${CMD} training.model.num_layers=4 training.model.num_heads=8 training.model.num_cls_layers=2 \
                 training.model.embed_dims='[512,1024,512]' training.model.pair_embed_dims='[128,128,128]'"
        fi

        echo "Submitting: ${JOB_NAME}"
        eval $CMD
    done
done

echo "Scaling study submitted. Monitor status with 'squeue -u $USER'"
