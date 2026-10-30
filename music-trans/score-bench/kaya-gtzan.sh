#!/bin/bash
#SBATCH --job-name=amt-gtzan-paper
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100-32gb:1
#SBATCH --mem=64G
#SBATCH --time=00:20:00
set -euo pipefail
umask 077
: "${BENCH_ROOT:?Set BENCH_ROOT to the external benchmark working directory}"
: "${PYTHON_BIN:?Set PYTHON_BIN to the isolated SheetSage2 interpreter}"
: "${MODEL_ASSETS:?Set MODEL_ASSETS to the downloaded public model bundles}"
ROOT="$BENCH_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
export HF_MODULES_CACHE="$ROOT/hf-modules"
exec "$PYTHON_BIN" -s "$ROOT/run_sheet_gtzan.py" --assets "$MODEL_ASSETS" --data "$ROOT/gtzan" --output "$ROOT/sheet-gtzan"
