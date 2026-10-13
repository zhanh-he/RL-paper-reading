#!/usr/bin/env bash
#SBATCH --job-name=wsb-quality
#SBATCH --account=ems011
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --nice=0
set -euo pipefail
work=/scratch/ems011/zhe/wsb-objective-20261013
export HF_HOME=/scratch/ems011/zhe/yue2-formal/hf-cache
export TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$work/cache/xdg"
python=/scratch/ems011/zhe/yue2-formal/.venv/bin/python
exec "$python" -u "$work/code/score_saved_quality.py" --bundle "$work/wsb_metrics" \
  --input "$work/input" --output "$work/quality-audit" --extras-python "$work/extras/bin/python" \
  --songeval /scratch/ems011/zhe/yue2-formal/SongEval
