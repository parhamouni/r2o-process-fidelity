#!/bin/bash
#SBATCH --job-name=r2o_taskD
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/taskD_%j.out
# Usage: export R2O_GDGB_ROOT=/path/to/data; sbatch scoring/slurm/taskD.sh   (run from scoring/)
cd "$(dirname "$0")/.." && mkdir -p slurm_logs
exec python taskD_timing_floor.py
