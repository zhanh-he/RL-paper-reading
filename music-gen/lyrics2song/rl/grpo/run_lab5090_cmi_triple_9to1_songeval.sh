#!/usr/bin/env bash
set -euo pipefail

root=/home/mengh/research/YuE2-posttrain
output="$root/outputs/cmi_triple_9to1_songeval_lr2e-5_train_20261007_lab5090"
manifest="$root/examples/cmi-pref-triple-text-lyrics-9to1-v2.json"

cd "$root"
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/home/mengh/.cache/huggingface

exec flock -n "$output.lock" .venv/bin/python -u examples/yue2_songeval_formal_9to1.py \
  --model "$root/models/YuE2-3B" \
  --vae "$root/models/YuE2-Vae" \
  --dataset-manifest "$manifest" \
  --expected-dataset-sha256 c39d43e81793bee3f312c7028a0483b5da41068db4544acd258115043101e46f \
  --reward-backend songeval \
  --songeval /home/mengh/research/SongEval-audit \
  --scorer-python "$root/.venv/bin/python" \
  --output "$output" \
  --max-steps 270 \
  --max-tokens 600 \
  --learning-rate 2e-5 \
  --kl-beta 0.01 \
  --compute-dtype bf16 \
  --checkpoint-every 25
