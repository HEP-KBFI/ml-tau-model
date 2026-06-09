#!/bin/bash

# Submission script to launch the five basic ParTau models to the SLURM batch system.
# Usage: ./submit_jobs.sh

# Ensure git status is clean to guarantee reproducibility
if [[ -n $(git status --porcelain) ]]; then
  echo "Error: Git status is not clean. Please commit or stash your changes before submitting jobs."
  exit 1
fi

# Get the current git revision
TAG=0609
GIT_REV=$(git rev-parse --short HEAD)
echo "Current Git Revision: $GIT_REV"

echo "Submitting SingleParTau jobs..."

# 1. SingleParTau: Tau ID (Tagging)
# sbatch --job-name=tauid train-gpu.sh \
#     training.model.name=SingleParTau \
#     training.model.task=is_tau \
#     output_dir=outputs/${TAG}_single_tauid_${GIT_REV}
# 
# # 2. SingleParTau: Charge ID
# sbatch --job-name=charge train-gpu.sh \
#     training.model.name=SingleParTau \
#     training.model.task=charge \
#     output_dir=outputs/${TAG}_single_charge_${GIT_REV}
# 
# # 3. SingleParTau: Decay Mode Classification
# sbatch --job-name=decaymode train-gpu.sh \
#     training.model.name=SingleParTau \
#     training.model.task=decay_mode \
#     output_dir=outputs/${TAG}_single_decaymode_${GIT_REV}
# 
# # 4. SingleParTau: Kinematics Regression
# sbatch --job-name=kinematics train-gpu.sh \
#     training.model.name=SingleParTau \
#     training.model.task=kinematics \
#     output_dir=outputs/${TAG}_single_kinematics_${GIT_REV}

echo "Submitting MultiParTau job..."

# 5. MultiParTau: Full Multi-task Model (with PCGrad)
sbatch --job-name=multipartau train-gpu.sh \
    training.model.name=MultiParTau \
    output_dir=outputs/${TAG}_multipartau_full_${GIT_REV}

echo "All jobs submitted. Monitor status with 'squeue -u $USER'"
