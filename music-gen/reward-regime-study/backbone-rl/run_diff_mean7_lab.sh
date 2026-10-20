#!/usr/bin/env bash
set -euo pipefail
study=/storage/zhanh_storage/songbench-priority-20261020
native=/storage/zhanh_storage/open-music-rl-20261014
deps=/storage/zhanh_storage/framework-smoke-20261015
code="$study/code"
output="$study/outputs/diff-mean7-formal"
mkdir -p "$output"
exec 9>"$study/diff-mean7.lock"
flock -n 9 || { echo 'Mean7 Diff job already owned'; exit 1; }
export CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
export PYTHONPATH="$deps/peft014:$deps/miles-deps:/storage/zhanh_storage/backbone-rl-20261015/sources/miles:$code"
export NLTK_DATA="$native/nltk_data" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
metrics=/home/mengh/research/WildSongBench-eval/wsb_metrics
"$metrics/.venvs/metrics/bin/python" -u "$code/reward_server.py" --bundle "$metrics" \
  --deps /home/mengh/research/reward-regimes/deps --mean7-only \
  --separator /home/mengh/.cache/torch/hub/checkpoints/955717e8-8726e21a.th \
  --receipt-dir "$output/scorer" --port 49330 > "$output/scorer.log" 2>&1 &
scorer=$!
cleanup(){ kill "$scorer" 2>/dev/null || true; wait "$scorer" 2>/dev/null || true; }
trap cleanup EXIT
ready=false
for _ in $(seq 1 120); do
  if curl -fsS http://127.0.0.1:49330/health >/dev/null 2>&1; then ready=true; break; fi
  kill -0 "$scorer" || exit 1
  sleep 5
done
[[ "$ready" == true ]] || exit 1
"$native/envs/diffrhythm2/bin/python" -u "$code/run_diff_miles.py" \
  --assets "$native/assets/diffrhythm2" --source "$native/sources/diffrhythm2" \
  --data "$native/data.json" --output "$output" --seconds 60 --steps 270 --rewards mean7 \
  --scorer http://127.0.0.1:49330 > "$output/train.log" 2>&1
