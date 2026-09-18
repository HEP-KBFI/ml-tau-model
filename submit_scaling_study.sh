#!/bin/bash
# Submitter for the ParTauDETR scaling study on LUMI.
#
#   ./submit_scaling_study.sh                                  # submit both ladders
#   STAGES=1 ./submit_scaling_study.sh                         # signal ladder only
#   DRY_RUN=1 STAGES=1 ./submit_scaling_study.sh               # print, submit nothing
#   MAX_EPOCHS=150 ./submit_scaling_study.sh                   # different budget
#   OUT_ROOT=/scratch/project_465001293/TauTr-scaling ./submit_scaling_study.sh
#   ./submit_scaling_study.sh training.trainer.limit_val_batches=20
#                                                              # trailing args -> train script
#
# One sbatch per run, in a loop. There is no manifest and no job array: each job
# carries its own configuration in its environment, so nothing has to agree
# about task numbering and a single run is resubmitted by rerunning this script
# with the ladders trimmed.
#
# Deliberately NOT `set -e`. A guard like `[[ -n "$X" ]] && Y=1` returns 1 when
# the test fails, which under `set -e` silently terminates the script -- that
# exact line, with an empty THROTTLE, is why an earlier version submitted
# nothing at all.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

# ---------------------------------------------------------------- the grid ---
# Doubling ladders anchored at the statistics available: 5M signal, 30M
# background. Each point halves the previous one, so every step asks the same
# question -- what does doubling this sample buy?
#
#   stage 1  signal varied, background held at its maximum
#   stage 2  background varied, signal held at its maximum
#
# The full-statistics point is the anchor of both and is emitted once, in
# stage 1, which is why the background ladder runs one step deeper:
# 6 + 7 - 1 = 12 distinct configurations, x 3 seeds = 36 runs.
SIG_MAX=${SIG_MAX:-5000000}
BKG_MAX=${BKG_MAX:-30000000}
SIG_HALVINGS=${SIG_HALVINGS:-5}
BKG_HALVINGS=${BKG_HALVINGS:-6}
SEEDS=(${SEEDS:-1 2 3})

# Same number of passes over its own data for every point.
MAX_EPOCHS="${MAX_EPOCHS:-100}"
# Optional hard cap; when set it wins and the train script forces max_epochs=-1.
MAX_STEPS="${MAX_STEPS:-}"

STAGES="${STAGES:-1,2}"
OUT_ROOT="${OUT_ROOT:-/scratch/project_465001293/scaling_$(date +%Y%m%d)}"
DRY_RUN="${DRY_RUN:-0}"

# git rev only: `git status` on a Lustre working tree takes long enough to look
# like a hang, and it was only ever used for a cosmetic label.
GIT_REV="$(git rev-parse --short HEAD 2>/dev/null)"
if [[ -z "$GIT_REV" ]]; then
    GIT_REV="unknown"
fi

# "$@" inside submit() would be the FUNCTION's arguments, so the script's
# trailing Hydra overrides are captured here first.
EXTRA_ARGS=("$@")

mkdir -p "$OUT_ROOT" logs/scaling

wanted () {  # is stage $1 in STAGES?
    case ",${STAGES}," in
        *",$1,"*) return 0 ;;
        *) return 1 ;;
    esac
}

submit () {  # submit <stage> <n_sig> <n_bkg> <label>
    local stage="$1" n_sig="$2" n_bkg="$3" label="$4" seed run_name out_dir
    if ! wanted "$stage"; then
        return 0
    fi
    for seed in "${SEEDS[@]}"; do
        run_name="s${stage}_${label}_seed${seed}_${GIT_REV}"
        out_dir="${OUT_ROOT}/${run_name}"
        exports="ALL,STAGE=${stage},N_SIG=${n_sig},N_BKG=${n_bkg},SEED=${seed}"
        exports="${exports},RUN_NAME=${run_name},OUT_DIR=${out_dir}"
        exports="${exports},MAX_EPOCHS=${MAX_EPOCHS}"
        if [[ -n "$MAX_STEPS" ]]; then
            exports="${exports},MAX_STEPS=${MAX_STEPS}"
        fi

        if [[ "$DRY_RUN" == "1" ]]; then
            printf '  [dry] %-42s sig=%-9s bkg=%-9s seed=%s\n' \
                "$run_name" "$n_sig" "$n_bkg" "$seed"
        else
            sbatch --job-name="$run_name" --export="$exports" \
                   train-scaling-lumi.sh ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
                | sed "s/^/  /"
        fi
        N_SUBMITTED=$((N_SUBMITTED + 1))
    done
}

echo "=============================================================="
echo " out_root   : $OUT_ROOT"
echo " stages     : $STAGES"
echo " max_epochs : $MAX_EPOCHS"
echo " max_steps  : ${MAX_STEPS:-none (epoch-budgeted)}"
echo " seeds      : ${SEEDS[*]}"
echo " git rev    : $GIT_REV"
if [[ $# -gt 0 ]]; then
    echo " extra args : $*"
fi
echo "=============================================================="

N_SUBMITTED=0

n_sig=$SIG_MAX
for (( i = 0; i <= SIG_HALVINGS; i++ )); do
    submit 1 "$n_sig" "$BKG_MAX" "sig${n_sig}"
    n_sig=$((n_sig / 2))
done

n_bkg=$BKG_MAX
for (( i = 0; i <= BKG_HALVINGS; i++ )); do
    if (( n_bkg != BKG_MAX )); then          # anchor already emitted in stage 1
        submit 2 "$SIG_MAX" "$n_bkg" "bkg${n_bkg}"
    fi
    n_bkg=$((n_bkg / 2))
done

echo "=============================================================="
if [[ "$DRY_RUN" == "1" ]]; then
    echo " ${N_SUBMITTED} run(s) would be submitted. Rerun without DRY_RUN=1."
else
    echo " ${N_SUBMITTED} run(s) submitted. squeue -u \$USER"
fi
