#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:l40:1
#SBATCH --mem-per-gpu 64G
#SBATCH -o logs/slurm-%x-%j-%N.out
#SBATCH --cpus-per-task=8
# Without this Slurm gives the job ONE cpu, so N dataloader workers all
# timeshare a single core and the first batch can take many minutes.
# Keep it >= training.dataloader.num_dataloader_workers + 2.
# 8 is measured, not guessed: featurisation needs ~1.5 cores on average
# (6.2 s per 102k jets, ~11M jets per epoch, ~455 s epochs) and bursts to
# ~6 when several workers build chunks at once. The run is GPU-bound, so
# more cores buy nothing and only lengthen the queue wait.

# Pass Hydra overrides as extra arguments, e.g.:
#
#   sbatch train-gpu-DETR.sh training.trainer.max_epochs=5
#   sbatch train-gpu-DETR.sh training.detr.num_queries=8
#   sbatch train-gpu-DETR.sh dataset.data_dir=/path/to/parquet
#

env | grep CUDA

# Slurm sets CUDA_VISIBLE_DEVICES to the node-global GPU index (e.g. "1" for the
# second GPU), but its cgroup already restricts this job to that GPU alone, so
# the process sees it as device 0. The stale index then points past the end of
# the visible set and CUDA reports no devices at all. Renumber densely from 0,
# but only when the indices really do exceed what is visible, so that nodes
# without cgroup device isolation keep their original mapping.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    n_visible=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
    max_idx=$(tr ',' '\n' <<<"$CUDA_VISIBLE_DEVICES" | sort -n | tail -1)
    if [[ "$max_idx" =~ ^[0-9]+$ ]] && [[ "$n_visible" -gt 0 ]] \
       && [[ "$max_idx" -ge "$n_visible" ]]; then
        export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((n_visible - 1)))"
        echo "Remapped CUDA_VISIBLE_DEVICES -> $CUDA_VISIBLE_DEVICES" \
             "($n_visible GPU(s) visible in this job's cgroup)"
    fi
fi

mkdir -p logs
# Per-job filename: concurrent jobs would otherwise all write logs/gpu_log.txt.
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
           --format=csv -l 10 > "logs/gpu_log-${SLURM_JOB_ID:-local}.txt" &
# Dataloader workers do elementwise numpy/awkward work, not BLAS, so extra
# threads per worker only contend for the cores the other workers need. PyTorch
# already calls set_num_threads(1) inside workers, but OMP/MKL/OpenBLAS are
# unset by default and would otherwise each spin up a full thread pool.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
./run.sh python3 mltau/scripts/train_ParTauDETR.py --config-name main_ParTauDETR "$@"
