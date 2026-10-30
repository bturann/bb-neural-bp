#!/usr/bin/env bash
# Submit only after selecting the cluster partition/account and installing the pinned environment.
#SBATCH --job-name=bbnbp-m6
#SBATCH --array=1234-1236
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-m6-%A-%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the codebase directory}"
exec .venv/bin/python scripts/m6.py train --device cuda --seed "${SLURM_ARRAY_TASK_ID}"
