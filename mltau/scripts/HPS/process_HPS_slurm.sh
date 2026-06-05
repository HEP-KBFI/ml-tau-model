#!/bin/bash

# Directories produced by split_parquet.py
# Update these paths to point to your chunk directories before running.

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

 ##################################
Z_BASE_NTUPLE_DIR="/home/laurits/tmp/ntupelized_tmp/p8_ee_Z_tautau_ecm91/"
Z_OUTPUT_DIR="/home/laurits/HPS/Z/"
Z_FILES=$(ls "$Z_BASE_NTUPLE_DIR")
xargs -n 20 <<< "$Z_FILES" | tr ' ' ',' | xargs -I {} sbatch mltau/scripts/HPS/submission.sh output_dir=$Z_OUTPUT_DIR files=[{}] base_ntuple_dir=$Z_BASE_NTUPLE_DIR

 ##################################
QQ_BASE_NTUPLE_DIR="/home/laurits/tmp/ntupelized_tmp/p8_ee_Z_qq_ecm91/"
QQ_OUTPUT_DIR="/home/laurits/HPS/QQ/"
QQ_FILES=$(ls "$QQ_BASE_NTUPLE_DIR")
xargs -n 20 <<< "$QQ_FILES" | tr ' ' ',' | xargs -I {} sbatch mltau/scripts/HPS/submission.sh output_dir=$QQ_OUTPUT_DIR files=[{}] base_ntuple_dir=$QQ_BASE_NTUPLE_DIR