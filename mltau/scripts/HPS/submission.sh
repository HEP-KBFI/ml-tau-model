#!/bin/bash
#SBATCH -p main
#SBATCH --job-name=HPS_processing
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -o HPS_logs/slurm-%x-%j-%N.out

./run.sh python3 mltau/scripts/HPS/process_HPS.py "$@"