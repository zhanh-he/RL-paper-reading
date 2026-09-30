#!/usr/bin/env bash
set -euo pipefail

root=/home/mengh/research/YuE2-posttrain
output="$root/outputs/yue2_songeval_cross_musecritic_20260930"
mkdir -p "$output"
args=(
  --repo /home/mengh/research/vocal2accomp-muse/src/MuseCritic
  --model /home/mengh/research/musecritic_beat_eval/models/MuseCritic
  --max-new-tokens 512
)
for arm in 1e3 1e2; do
  run="$root/outputs/songeval_grpo_lr${arm}_20260930"
  for step in 000000 000001 000005 000025 000050 000100; do
    for index in 0 1 2; do
      audio="$run/step_$step/heldout_$index/audio.flac"
      result="$output/lr$arm/step_$step/heldout_$index"
      test -s "$audio"
      args+=(--pair "$audio" "$result")
    done
  done
done
export CUDA_VISIBLE_DEVICES=0
/home/mengh/miniconda3/envs/musecritic/bin/python -u \
  "$root/examples/score_musecritic_batch.py" "${args[@]}" >"$output/run.log" 2>&1
echo "$(date -Is) scored all 36 fixed SongEval-arm held-out clips" >"$output/status.log"
