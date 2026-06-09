#!/bin/bash

#keras is not used, but for some reason, it's imported somewhere and crashes if this is not specified
export KERAS_BACKEND=torch
apptainer exec --env PYTHONPATH=`pwd`:`pwd`/mltau --nv pytorch.simg\:2025-09-01 "$@"
