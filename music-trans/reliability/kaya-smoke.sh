#!/bin/bash
#SBATCH --job-name=amt-sheet-audit
#SBATCH --account=ems011
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100-32gb:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
set -euo pipefail
umask 077
ROOT="/group/ems011/zhe/amt-reliability-20261029"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_MODULES_CACHE="$ROOT/hf-modules"
export OMP_NUM_THREADS=4
exec /group/ems011/zhe/muse-public/teacher-overlay-dffbf77e/sheetsage2-py311-overlay/bin/python -s "$ROOT/code/run_sheetsage2.py" --assets /group/ems011/zhe/muse-public/assets --inputs "$ROOT/inputs" --output "$ROOT/output-first" --limit 48
