#!/bin/bash
# Array runner for the ParTauDETR scaling study. Submitted by
# submit_scaling_study.sh, which exports MANIFEST, OUT_ROOT and MAX_STEPS and
# sets --array.
#
# Not meant to be sbatch'ed directly: it reads its configuration from the
# manifest line matching $SLURM_ARRAY_TASK_ID.
#
# Resources are identical to train-gpu-lumi.sh so that timings are comparable
# across the study; see that file for why 7 cores / 60 G.
#SBATCH --job-name=scal
#SBATCH --account=project_465001293
# Sized from the largest configuration: ~24-36 h observed. small-g allows
# 3 days, so 48 h leaves margin without asking for the maximum (shorter
# requests schedule better on a shared partition).
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=small-g
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --no-requeue
#SBATCH -o logs/scaling/slurm-%A_%a.out

set -euo pipefail

cd /scratch/project_465001293/ml-tau-model
mkdir -p logs/scaling

: "${MANIFEST:?must be exported by submit_scaling_study.sh}"
: "${OUT_ROOT:?must be exported by submit_scaling_study.sh}"
: "${MAX_STEPS:?must be exported by submit_scaling_study.sh}"

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")
[[ -n "$LINE" ]] || { echo "no manifest line ${SLURM_ARRAY_TASK_ID}" >&2; exit 1; }
read -r STAGE N_SIG N_BKG SEED RUN_NAME <<<"$LINE"

OUT_DIR="${OUT_ROOT}/${RUN_NAME}"
mkdir -p "$OUT_DIR"

echo "=============================================================="
echo " run        : $RUN_NAME"
echo " stage      : $STAGE   task $SLURM_ARRAY_TASK_ID of $MANIFEST"
echo " n_sig      : $N_SIG"
echo " n_bkg      : $N_BKG"
echo " seed       : $SEED"
echo " max_steps  : $MAX_STEPS"
echo " output_dir : $OUT_DIR"
echo "=============================================================="

# See train-gpu-lumi.sh for the reasoning behind each of these.
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

# Comet's GPU metrics are NVIDIA-only (pynvml), so sample rocm-smi ourselves.
# rocm-smi has no repeat flag, hence the explicit loop.
if command -v rocm-smi >/dev/null 2>&1; then
    (
        while true; do
            printf '%s ' "$(date -Is)"
            rocm-smi --showuse --showmemuse --csv 2>&1 | tr '\n' ' '
            printf '\n'
            sleep 30
        done
    ) > "${OUT_DIR}/gpu_log.txt" 2>&1 &
    # Capture the pid now: $! inside the trap would be evaluated at exit time,
    # by which point it may name a different background job.
    ROCM_SMI_PID=$!
    trap 'kill ${ROCM_SMI_PID} 2>/dev/null' EXIT
fi

# Written BEFORE training so a crashed or timed-out run still records what it
# was attempting. metrics.json (from the train script) carries the results.
cat > "${OUT_DIR}/run_meta.json" <<JSON
{
  "run_name": "${RUN_NAME}",
  "stage": "${STAGE}",
  "n_sig_requested": ${N_SIG},
  "n_bkg_requested": ${N_BKG},
  "seed": ${SEED},
  "max_steps": ${MAX_STEPS},
  "slurm_job": "${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-}",
  "partition": "${SLURM_JOB_PARTITION:-}",
  "cpus_per_task": ${SLURM_CPUS_PER_TASK:-0},
  "gpu": "$(command -v rocm-smi >/dev/null 2>&1 && rocm-smi --showproductname --csv 2>/dev/null | tail -1 || echo unknown)"
}
JSON

# One seed drives both axes of randomness: which jets are selected
# (dataset.selection_seed) and how they are trained on (training.seed). That is
# the realistic run-to-run spread the 3 repeats are meant to measure.
#
# max_steps forces max_epochs=-1 inside the train script, so every run gets the
# same optimizer budget regardless of dataset size, and OneCycleLR anneals over
# exactly that many steps. Note the epoch COUNT therefore varies ~200x across
# the grid; metrics.json records it.
SECONDS=0
./run-lumi.sh python3 mltau/scripts/train_ParTauDETR.py \
    --config-name main_ParTauDETR \
    output_dir="$OUT_DIR" \
    dataset.max_jets_per_sample.z="$N_SIG" \
    dataset.max_jets_per_sample.qq="$N_BKG" \
    dataset.selection_seed="$SEED" \
    training.seed="$SEED" \
    training.trainer.max_steps="$MAX_STEPS" \
    logging.comet.experiment_name="$RUN_NAME" \
    "logging.comet.tags=[scaling,stage-${STAGE}]" \
    "$@"

echo "finished ${RUN_NAME} in ${SECONDS} s"

# Whole-job wall time, deliberately distinct from metrics.json's wall_seconds,
# which times fit() only and excludes container startup and the row-group
# metadata scan. Guarded: bookkeeping must not fail the job under `set -e`.
if command -v python3 >/dev/null 2>&1; then
    python3 - "$OUT_DIR" "$SECONDS" <<'INNER' || echo "warning: job_wall_seconds not recorded"
import json, pathlib, sys
meta_path = pathlib.Path(sys.argv[1]) / "run_meta.json"
meta = json.loads(meta_path.read_text())
meta["job_wall_seconds"] = int(sys.argv[2])
meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
print(f"recorded job_wall_seconds={meta['job_wall_seconds']} in {meta_path}")
INNER
else
    echo "warning: no host python3; job_wall_seconds not recorded"
fi
