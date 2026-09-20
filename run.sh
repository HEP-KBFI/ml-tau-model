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
# training module does not import without it. Resolution order: MLTAU_DATA_DIR,
# the git submodule (populated by `git submodule update --init`), a sibling
# checkout next to this repository.
pythonpath="$repo_dir:$repo_dir/mltau"
mltau_data_dir=""
for candidate in "${MLTAU_DATA_DIR:-}" "$repo_dir/ml-tau-data" "$repo_dir/../ml-tau-data"; do
	if [[ -n "$candidate" && -d "$candidate/ntupelizer" ]]; then
		mltau_data_dir="$(cd -- "$candidate" && pwd)"
		break
	fi
done
if [[ -n "$mltau_data_dir" ]]; then
	pythonpath="$pythonpath:$mltau_data_dir"
else
	echo "run.sh: ml-tau-data not found (run 'git submodule update --init' or set MLTAU_DATA_DIR); decay-mode code will not import" >&2
fi

apptainer exec -B /scratch/persistent,/local,/home --env PYTHONPATH="$pythonpath" --nv /home/software/singularity/pytorch.simg\:2025-09-01 "$@"
