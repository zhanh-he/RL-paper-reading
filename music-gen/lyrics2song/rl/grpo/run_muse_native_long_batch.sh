#!/usr/bin/env bash
set -euo pipefail
root=/home/mengh/research/Muse-posttrain
study="$root/outputs/cmi_formal_recovery_20261012"
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo 'GPU occupied; leave other jobs untouched' >&2
  exit 3
fi
python=/home/mengh/miniconda3/envs/muse-grpo/bin/python
args=(--study "$study" --valid "$study/muse-cmi-triple-valid.jsonl"
      --model /home/mengh/research/vocal2accomp-muse/models/Muse-0.6b)
"$python" -u "$root/infer/generate_muse_formal_batch.py" generate "${args[@]}"
mapfile -t tokens < <(find "$study/native_vllm" "$study"/*_train_v2/long_replay_vllm \
  -name audio.json -type f | sort | while IFS= read -r path; do
    [[ -f "${path%/audio.json}/failure.json" ]] || printf '%s\n' "$path"
  done)
[[ ${#tokens[@]} -gt 0 ]]
/home/mengh/miniconda3/envs/mucodec-cu128/bin/python -u "$root/infer/decode_muse_tokens.py" \
  --mucodec /home/mengh/research/MuCodec-posttrain --duration 40.96 --steps 20 \
  --token-file "${tokens[@]}"
"$python" -u "$root/infer/generate_muse_formal_batch.py" finalize "${args[@]}"
