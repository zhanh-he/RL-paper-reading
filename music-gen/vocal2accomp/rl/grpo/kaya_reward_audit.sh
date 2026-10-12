#!/bin/bash
#SBATCH --job-name=vocal-bcr-audit
#SBATCH --partition=work
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/group/ems011/zhe/vocal-musdb-20261012/logs/audit-%j.out
#SBATCH --error=/group/ems011/zhe/vocal-musdb-20261012/logs/audit-%j.err
set -euo pipefail
module load ffmpeg/8.0.1
ROOT=/group/ems011/zhe/vocal-musdb-20261012
PYTHON=/group/ems011/zhe/envs/auto-structbeat/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
cd "$ROOT/code/rl/grpo"
"$PYTHON" -m unittest test_train_lada_band test_richness_v0 test_formal_protocol test_prepare_musdb
"$PYTHON" "$ROOT/code/rewards/audit_formal.py" \
  --reward-root /group/ems011/zhe/vocal2accomp-muse --output "$ROOT/reward-audit"
touch "$ROOT/reward-audit/audit.complete"
