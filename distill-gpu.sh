#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:l40:1
#SBATCH --mem-per-gpu 64G
#SBATCH -o logs/slurm-%x-%j-%N.out

env | grep CUDA
nvidia-smi -L
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
./run.sh python3 mltau/scripts/distill.py "$@"
