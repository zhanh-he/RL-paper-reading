#!/usr/bin/env bash
#SBATCH --job-name=wsb-prepare
#SBATCH --account=ems011
#SBATCH --partition=work
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --nice=0
set -euo pipefail
export EVAL_WORK=/scratch/ems011/zhe/wsb-objective-20261013
export BASE_PYTHON=/scratch/ems011/zhe/yue2-formal/.venv/bin/python
export HF_HOME=/scratch/ems011/zhe/yue2-formal/hf-cache
export PREPARE_METRICS=songbench,mulan,allmusiccaps
bash "$EVAL_WORK/code/prepare_cluster.sh"
