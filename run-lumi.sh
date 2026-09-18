#!/bin/bash

cd /scratch/project_465001293/ml-tau-model

# module load LUMI/24.03 partition/G

export IMG=/scratch/project_465001293/2026-09_pytorch_rocm.simg
export PYTHONPATH=hep_tfds
export MIOPEN_USER_DB_PATH=/tmp/${USER}-${SLURM_JOB_ID}-miopen-cache
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
export TF_CPP_MAX_VLOG_LEVEL=-1  # to suppress ROCm fusion is enabled messages
export ROCM_PATH=/opt/rocm
export KERAS_BACKEND=torch
# Slurm redirects stdout to a file, which makes it block-buffered. Plain
# print() output (the dataset/row-group lines) then sits in the buffer for
# a long time while Lightning's logging-based messages appear immediately,
# making a healthy run look hung.
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}

# Defaults for settings the sbatch wrappers also set. Defined here with :- so
# that a bare `./run-lumi.sh python3 ...` -- a smoke test, an evaluation, an
# interactive session -- gets the same environment as a batch job, while an
# explicit export in the wrapper still wins. Previously these lived only in
# train-gpu-lumi.sh / train-scaling-lumi.sh, so any direct invocation silently
# ran with different thread and allocator settings.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
# PYTORCH_HIP_ALLOC_CONF is left unset by default: expandable_segments
# hung on the first device allocation on ROCm. Export it explicitly to
# opt back in.
#export MIOPEN_DISABLE_CACHE=true
#export NCCL_DEBUG=INFO
#export MIOPEN_ENABLE_LOGGING=1
#export MIOPEN_ENABLE_LOGGING_CMD=1
#export MIOPEN_LOG_LEVEL=4

# env
#TF training
# Deliberately NOT --rocm, and deliberately no explicit -B /dev/kfd -B /dev/dri.
#
# --rocm injects the HOST's ROCm/graphics libraries into /.singularity.d/libs.
# LUMI's host libdrm.so.2 is linked against GLIBC 2.38, this image is Ubuntu
# 22.04 (glibc 2.35), so `import torch` dies with
#   ImportError: ... version `GLIBC_2.38' not found (required by libdrm.so.2)
# Apptainer appends its injected directory AFTER whatever LD_LIBRARY_PATH is
# set below, so the export does not protect against this.
#
# Nothing from the host is needed: the image carries a complete ROCm 6.2.3, and
# the kernel driver is reached through /dev/kfd, /dev/dri and /sys, all of which
# Apptainer mounts by default. Binding those paths EXPLICITLY shadows the
# default mounts and makes the GPU disappear ("No HIP GPUs are available").
#
# Verified on LUMI (job 22077644, MI250X): this form reports device_count 1 and
# a correct first allocation; --rocm with an untouched LD_LIBRARY_PATH fails at
# import; --rocm with the container's lib dirs ahead of /.singularity.d/libs
# also works but depends on that ordering never being disturbed.
singularity exec \
    -B /scratch/project_465001293 \
    -B /tmp \
    --env PYTHONPATH="`pwd`:`pwd`/mltau:${MLTAU_DATA_DIR:-`pwd`/../ml-tau-data}" \
    --env LD_LIBRARY_PATH=/opt/rocm/lib/ \
    --env CUDA_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES \
     $IMG "$@"
