#!/usr/bin/env bash
set -euo pipefail
study=/storage/zhanh_storage/backbone-rl-20261015
native=/storage/zhanh_storage/open-music-rl-20261014
deps=/storage/zhanh_storage/framework-smoke-20261015
exec 9>"$study/diff-training.lock"
flock -n 9 || { echo 'This study already owns a DiffRhythm trainer'; exit 1; }
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$deps/peft014:$deps/miles-deps:$study/sources/miles:$study/code"
export NLTK_DATA="$native/nltk_data" HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "$native/envs/diffrhythm2/bin/python" -u "$study/code/run_diff_miles.py" \
  --assets "$native/assets/diffrhythm2" --source "$native/sources/diffrhythm2" \
  --data "$native/data.json" --output "$study/diff-formal" --seconds 60 --steps 270 \
  --scorer http://127.0.0.1:43140 "$@"
