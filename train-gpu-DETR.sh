#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:rtx
#SBATCH --mem-per-gpu 40G
#SBATCH -o logs/slurm-%x-%j-%N.out

# Pass Hydra overrides as extra arguments, e.g.:
#
#   sbatch train-gpu-DETR.sh training.trainer.max_epochs=5
#   sbatch train-gpu-DETR.sh training.detr.num_queries=8
#   sbatch train-gpu-DETR.sh dataset.data_dir=/path/to/parquet
#

env | grep CUDA
nvidia-smi -L
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
./run.sh python3 mltau/scripts/train_ParTauDETR.py --config-name main_ParTauDETR "$@"
#SBATCH --gres gpu:l40:1
#SBATCH --mem-per-gpu 64G
