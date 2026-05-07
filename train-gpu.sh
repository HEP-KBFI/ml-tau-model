#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:l40:1
#SBATCH --mem-per-gpu 64G
#SBATCH -o logs/slurm-%x-%j-%N.out

# To select the model, pass Hydra overrides as extra arguments, e.g.:
#
#   sbatch train-gpu.sh training.model.name=MultiParTau
#
#   sbatch train-gpu.sh training.model.name=SingleParTau training.model.task=is_tau
#   sbatch train-gpu.sh training.model.name=SingleParTau training.model.task=charge
#   sbatch train-gpu.sh training.model.name=SingleParTau training.model.task=decay_mode
#   sbatch train-gpu.sh training.model.name=SingleParTau training.model.task=kinematics

env | grep CUDA
nvidia-smi -L
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
./run.sh python3 mltau/scripts/train.py "$@"