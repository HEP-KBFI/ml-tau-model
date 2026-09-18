#!/bin/bash

#keras is not used, but for some reason, it's imported somewhere and crashes if this is not specified
export KERAS_BACKEND=torch

# The following lines make the mltau module accessible from Jupyter notebook
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_dir" || exit 1
if [[ "$1" == "jupyter" && "$2" == "server" ]]; then
	jupyter_root_dir="${JUPYTER_ROOT_DIR:-$repo_dir/mltau/notebooks}"
	set -- "$1" "$2" "--ServerApp.root_dir=$jupyter_root_dir" "${@:3}"
fi

# ml-tau-data provides ntupelizer.tools.tau_decaymode, the single definition of
# the tau decay mode; mltau.tools.evaluation.set_to_set_models imports it, so the
# training module does not import without it. Default: a sibling checkout.
mltau_data_dir="${MLTAU_DATA_DIR:-$repo_dir/../ml-tau-data}"
pythonpath="$repo_dir:$repo_dir/mltau"
if [[ -d "$mltau_data_dir" ]]; then
	pythonpath="$pythonpath:$(cd -- "$mltau_data_dir" && pwd)"
else
	echo "run.sh: ml-tau-data not found at $mltau_data_dir (set MLTAU_DATA_DIR); decay-mode code will not import" >&2
fi

apptainer exec -B /scratch/persistent,/local,/home --env PYTHONPATH="$pythonpath" --nv /home/software/singularity/pytorch.simg\:2025-09-01 "$@"
