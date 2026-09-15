#!/bin/bash

#keras is not used, but for some reason, it's imported somewhere and crashes if this is not specified
export KERAS_BACKEND=torch

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_dir" || exit 1

if [[ "$1" == "jupyter" && "$2" == "server" ]]; then
	jupyter_root_dir="${JUPYTER_ROOT_DIR:-$repo_dir/mltau/notebooks}"
	set -- "$1" "$2" "--ServerApp.root_dir=$jupyter_root_dir" "${@:3}"
fi

apptainer exec -B /scratch/persistent,/local,/home/${USER} --env PYTHONPATH="$repo_dir:$repo_dir/mltau" --nv /home/software/singularity/pytorch.simg\:2025-09-01 "$@"
