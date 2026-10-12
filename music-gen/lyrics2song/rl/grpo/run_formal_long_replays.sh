#!/usr/bin/env bash
set -euo pipefail

root=${YUE2_ROOT:-/home/mengh/research/YuE2-posttrain}
reward=${REWARD:-songeval}
case "$reward" in songeval|musecritic) ;; *) exit 2 ;; esac
baseline="$root/outputs/cmi_triple_9to1_songeval_lr3e-4_train_20261007_lab5090/long_replay/step_000270/baseline"
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME=/home/mengh/.cache/huggingface
cd "$root"
for lr in 2e-5 1e-4 3e-4; do
  if [[ "$reward" == songeval ]]; then
    run="$root/outputs/cmi_triple_9to1_${reward}_lr${lr}_train_20261007_lab5090"
  else
    run="$root/outputs/cmi_triple_9to1_${reward}_lr${lr}_train_20261006"
  fi
  for step in 1 25 50 100 270; do
    if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
      echo "GPU occupied; stopped before $reward LR=$lr step=$step" >&2
      exit 3
    fi
    echo "$(date -Iseconds) $reward LR=$lr step=$step"
    .venv/bin/python -u examples/render_yue2_long_replay.py \
      --run-root "$run" --dataset "$root/examples/cmi-pref-triple-text-lyrics-9to1-v2.json" \
      --model "$root/models/YuE2-3B" --vae "$root/models/YuE2-Vae" \
      --step "$step" --max-tokens 1600 --min-tokens 200 --baseline-root "$baseline"
  done
done
