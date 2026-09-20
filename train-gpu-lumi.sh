#!/bin/bash
#SBATCH --job-name=mltau
#SBATCH --account=project_465001293
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=small-g
#SBATCH --gpus-per-task=1
# LUMI-G runs in low-noise mode: 8 of the 64 cores are reserved (the first core
# of each CCD), leaving 56 usable across 8 GCDs, i.e. 7 per GCD. Billing is per
# slice of 8 cores / 64 GB per GCD, so 7 cores and 60 GB cost exactly the same
# as 4 cores and 40 GB -- there is no reason to ask for less.
#
# 7 is sized from measurement, not guessed: featurisation needs ~1.5 cores on
# average (6.2 s per 102k jets against ~455 s epochs) and bursts to ~6 when
# several dataloader workers build chunks at once. Training is GPU-bound, so
# this only has to keep the GPU fed.
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --no-requeue
#SBATCH -o logs/slurm-%x-%j-%N.out

# Pass Hydra overrides as extra arguments, e.g.:
#
#   sbatch train-gpu-lumi.sh training.trainer.max_steps=20000
#   sbatch train-gpu-lumi.sh dataset.max_jets_per_sample.qq=5000000
#   sbatch train-gpu-lumi.sh dataset.data_dir=/scratch/project_465001293/data
#

cd /scratch/project_465001293/ml-tau-model
mkdir -p logs

# Dataloader workers do elementwise numpy/awkward work, not BLAS, so extra
# threads per worker only contend for the cores the other workers need. PyTorch
# already calls set_num_threads(1) inside workers, but OMP/MKL/OpenBLAS are
# unset by default and would otherwise each spin up a full thread pool.
# Without this, plain print() output is block-buffered into the Slurm log
# and a running job looks hung.
export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# NOTE: deliberately NOT setting PYTORCH_HIP_ALLOC_CONF=expandable_segments:True.
# That was ported from the CUDA launcher, where the equivalent
# PYTORCH_CUDA_ALLOC_CONF setting is well exercised. On ROCm it uses the virtual
# memory mapping path and was observed to hang on the FIRST device allocation
# (torch.cuda.is_available() and get_device_name() still succeed, because
# neither allocates). Re-enable only if fragmentation becomes a real problem,
# and re-test the first allocation when you do:
#   sbatch --export=ALL,PYTORCH_HIP_ALLOC_CONF=expandable_segments:True ...

# Comet's GPU system metrics are NVIDIA-only (pynvml, keyed off
# CUDA_VISIBLE_DEVICES), so the sys.gpu.* panels stay empty on LUMI. Sample
# rocm-smi instead. Per-job filename: concurrent jobs must not share it.
if command -v rocm-smi >/dev/null 2>&1; then
    # rocm-smi has no repeat flag (unlike `nvidia-smi -l`), so loop by hand.
    (
        while true; do
            printf '%s ' "$(date -Is)"
            rocm-smi --showuse --showmemuse --csv 2>&1 | tr '\n' ' '
            printf '\n'
            sleep 10
        done
    ) > "logs/gpu_log-${SLURM_JOB_ID:-local}.txt" 2>&1 &
    ROCM_SMI_PID=$!
    trap 'kill ${ROCM_SMI_PID} 2>/dev/null' EXIT
else
    echo "rocm-smi not found outside the container; skipping GPU sampling"
fi

env | grep -E "ROCR|HIP|CUDA_VISIBLE" || true

# No CUDA_VISIBLE_DEVICES remap here, unlike train-gpu-DETR.sh. That fix exists
# because Slurm hands out node-global NVIDIA indices while the cgroup renumbers
# from 0; on LUMI run-lumi.sh already maps ROCR_VISIBLE_DEVICES into
# CUDA_VISIBLE_DEVICES for the container, and nvidia-smi does not exist here.
./run-lumi.sh python3 mltau/scripts/train_ParTauDETR.py \
    --config-name main_ParTauDETR "$@"
