#!/bin/bash

# Submit scratch and staged-distillation Mixer training for all four tasks.
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

declare -A TEACHER_CKPTS=(
    ["is_tau"]="outputs/0612_single_tauid_b8483f6/models/ParT-model_best.ckpt"
    ["charge"]="outputs/0612_single_charge_b8483f6/models/ParT-model_best.ckpt"
    ["decay_mode"]="outputs/0612_single_decaymode_b8483f6/models/ParT-model_best.ckpt"
    ["kinematics"]="outputs/0612_single_kinematics_b8483f6/models/ParT-model_best.ckpt"
)

for TASK in "${TASKS[@]}"; do
    TEACHER_CKPT="${TEACHER_CKPTS[$TASK]}"
    if [[ ! -f "$TEACHER_CKPT" ]]; then
        echo "Error: teacher checkpoint not found for $TASK: $TEACHER_CKPT"
        exit 1
    fi
done

for TASK in "${TASKS[@]}"; do
    echo "Submitting STANDARD Mixer training for task: $TASK"
    sbatch --job-name="mix_${TASK}" train-gpu.sh \
        training.model.name=SingleParTau \
        training.model.backbone=Mixer \
        training.model.task="$TASK" \
        dataset.sort_by_pt=true \
        output_dir=outputs/${TAG}_standard_${TASK}_${GIT_REV}

    echo "Submitting staged distillation for task: $TASK"
    TEACHER_CKPT="${TEACHER_CKPTS[$TASK]}"
    sbatch --job-name="dst_${TASK}" distill-gpu.sh \
        training.model.name=SingleParTau \
        training.model.backbone=Mixer \
        training.model.task="$TASK" \
        dataset.sort_by_pt=true \
        teacher_checkpoint="$(pwd)/$TEACHER_CKPT" \
        output_dir=outputs/${TAG}_distill_${TASK}_${GIT_REV}
done

echo "All eight Mixer jobs submitted. Monitor with: squeue -u $USER"
