#!/bin/bash
#SBATCH --job-name=r2o_taskB
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/taskB_%j.out
# Usage: export R2O_GDGB_ROOT=/path/to/data; sbatch scoring/slurm/taskB.sh   (run from scoring/)
cd "$(dirname "$0")/.." && mkdir -p slurm_logs
exec python taskB_count_conditional.py
