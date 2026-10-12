#!/usr/bin/env bash
set -euo pipefail
root=/home/mengh/research/Muse-posttrain
study="$root/outputs/cmi_formal_recovery_20261012"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
python=/home/mengh/miniconda3/envs/muse-grpo/bin/python
args=(--model /home/mengh/research/vocal2accomp-muse/models/Muse-0.6b
  --valid "$study/muse-cmi-triple-valid.jsonl"
  --mucodec /home/mengh/research/MuCodec-posttrain
  --codec-python /home/mengh/miniconda3/envs/mucodec-cu128/bin/python
  --decoder "$root/infer/decode_muse_tokens.py")
idle() {
  if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
    echo 'GPU occupied; leave other jobs untouched' >&2
    exit 3
  fi
}
idle
"$python" -u "$root/infer/render_muse_formal_long.py" "${args[@]}" --output "$study/long_baseline"
for reward in musecritic songeval; do
  for lr in 1e-6 3e-6 1e-5; do
    run="$study/${reward}_lr${lr}_train_v2"
    mapfile -t checkpoints < <(find "$run/checkpoint" -name adapter_model.safetensors | sort -V | tac)
    for file in "${checkpoints[@]}"; do
      adapter=${file%/adapter_model.safetensors}
      step=${adapter##*/checkpoint-}
      case "$step" in 30|60|90|120|270) ;; *) continue ;; esac
      idle
      echo "$(date -Iseconds) Muse $reward LR=$lr step=$step"
      "$python" -u "$root/infer/render_muse_formal_long.py" "${args[@]}" \
        --adapter "$adapter" --output "$run/long_replay/step_$(printf '%06d' "$step")/grpo"
    done
  done
done
