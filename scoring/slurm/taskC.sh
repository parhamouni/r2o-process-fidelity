#!/bin/bash
#SBATCH --job-name=r2o_taskC
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/taskC_%j.out
# Usage: export R2O_GDGB_ROOT=/path/to/data; sbatch scoring/slurm/taskC.sh   (run from scoring/)
cd "$(dirname "$0")/.." && mkdir -p slurm_logs
exec python taskC_rr_band.py
