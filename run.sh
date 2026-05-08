#!/bin/bash

#keras is not used, but for some reason, it's imported somewhere and crashes if this is not specified
export KERAS_BACKEND=torch
apptainer exec -B /scratch/persistent,/local,/home/laurits/,/home/norman --env PYTHONPATH=`pwd`:`pwd`/mltau --nv /home/software/singularity/pytorch.simg\:2025-09-01 "$@"