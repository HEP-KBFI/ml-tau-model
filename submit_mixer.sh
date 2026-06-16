#!/bin/bash

# Submission script to launch the MLPMixer models (standard and distillation) to the SLURM batch system.
# Usage: ./submit_mixer.sh

# Ensure git status is clean to guarantee reproducibility
if [[ -n $(git status --porcelain) ]]; then
  echo "Error: Git status is not clean. Please commit or stash your changes before submitting jobs."
  exit 1
fi

# Get the current git revision
TAG="$(date +%m%d)_mixer"
GIT_REV=$(git rev-parse --short HEAD)
echo "Current Git Revision: $GIT_REV"

TASKS=("is_tau" "charge" "decay_mode" "kinematics")

declare -A TEACHER_CKPTS
TEACHER_CKPTS=(
    ["is_tau"]="outputs/0612_single_tauid_b8483f6/models/ParT-model_best.ckpt"
    ["charge"]="outputs/0612_single_charge_b8483f6/models/ParT-model_best.ckpt"
    ["decay_mode"]="outputs/0612_single_decaymode_b8483f6/models/ParT-model_best.ckpt"
    ["kinematics"]="outputs/0612_single_kinematics_b8483f6/models/ParT-model_best.ckpt"
)

echo "Submitting MLPMixer jobs..."

for TASK in "${TASKS[@]}"; do
    echo "====================================================="
    echo "Submitting STANDARD training for task: $TASK"
    echo "====================================================="
    sbatch --job-name=mix_${TASK} train-gpu.sh \
        training.model.name=SingleParTau \
        training.model.backbone=Mixer \
        training.model.task="$TASK" \
        output_dir=outputs/${TAG}_standard_${TASK}_${GIT_REV}

    echo "====================================================="
    echo "Submitting DISTILLATION training for task: $TASK"
    echo "====================================================="
    TEACHER_CKPT="${TEACHER_CKPTS[$TASK]}"
    
    sbatch --job-name=dst_${TASK} distill-gpu.sh \
        training.model.name=SingleParTau \
        training.model.backbone=Mixer \
        training.model.task="$TASK" \
        +teacher_checkpoint="$(pwd)/$TEACHER_CKPT" \
        output_dir=outputs/${TAG}_distill_${TASK}_${GIT_REV}
done

echo "====================================================="
echo "All jobs submitted. Monitor status with 'squeue -u $USER'"
echo "====================================================="
