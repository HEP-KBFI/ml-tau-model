#!/bin/bash

# Submission script to launch distillation of MLP-Mixer from ParT teacher.
# Usage: ./submit_distillation.sh

# Ensure git status is clean to guarantee reproducibility
if [[ -n $(git status --porcelain) ]]; then
  echo "Error: Git status is not clean. Please commit or stash your changes before submitting jobs."
  exit 1
fi

# Get the current git revision
TAG=0612
GIT_REV=$(git rev-parse --short HEAD)
TEACHER_DIR="outputs/0612_single_tauid_b8483f6"
TEACHER_CKPT="${TEACHER_DIR}/models/ParT-model_best.ckpt"

if [[ ! -f "$TEACHER_CKPT" ]]; then
    echo "Error: Teacher checkpoint not found at $TEACHER_CKPT"
    exit 1
fi

echo "Current Git Revision: $GIT_REV"
echo "Teacher Checkpoint: $TEACHER_CKPT"

echo "Submitting Mixer Distillation jobs..."

# 1. Distill Mixer for Tau ID (Tagging)
sbatch --job-name=distill_tauid distill-gpu.sh \
    teacher_checkpoint=$TEACHER_CKPT \
    training.model.name=SingleParTau \
    training.model.task=is_tau \
    training.model.backbone=Mixer \
    training.model.embed_dim=128 \
    distill_alpha=0.5 \
    output_dir=outputs/${TAG}_distill_tauid_${GIT_REV}

echo "Job submitted. Monitor status with 'squeue -u $USER'"
