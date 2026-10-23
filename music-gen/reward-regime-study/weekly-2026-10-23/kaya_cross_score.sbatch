#!/usr/bin/env bash
#SBATCH --job-name=yue2-SE-SB-cross
#SBATCH --account=ems011
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --nice=0
set -euo pipefail
work=/scratch/ems011/zhe/progression-20261023
bundle=/scratch/ems011/zhe/wsb-objective-20261013/wsb_metrics
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
export PYTHONPATH="$work/code" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec "$bundle/.venvs/metrics/bin/python" -u "$work/code/score_existing_cross.py" \
  --input "$work/input/input.json" --output "$work/outputs/yue2-songeval3e4-songbench" \
  --bundle "$bundle" --deps /scratch/ems011/zhe/wsb-objective-20261013/extras/lib/python3.12/site-packages
