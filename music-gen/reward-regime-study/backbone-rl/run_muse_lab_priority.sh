#!/usr/bin/env bash
set -euo pipefail
study=/storage/zhanh_storage/songbench-priority-20261020
code="$study/code"
base=/home/mengh/research/vocal2accomp-muse
native=/home/mengh/research/vocal2accomp-muse-online
train=/home/mengh/miniconda3/envs/muse-grpo/bin
python="$train/python"
: "${REGIME_REWARD:?Explicit reward required}"
lr=${MUSE_LR:-3e-4}
lr_tag=${lr/-/}
output=${MUSE_OUTPUT:-"$study/outputs/muse-${REGIME_REWARD}-lr${lr_tag}-beta5e2-seed42"}
steps=${MUSE_STEPS:-270}
if [[ "$steps" != 270 ]]; then output="$study/acceptance/muse-${REGIME_REWARD}-${steps}steps-${MUSE_ATTEMPT:-1}"; fi
mkdir -p "$output"
exec 9>"$output/training.lock"
flock -n 9 || { echo 'Exact arm already owned'; exit 1; }
export CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false NPROC_PER_NODE=1 MASTER_PORT=49324
export MUSE_BASE_PLUGIN="$code/muse_native_plugin.py" MUSE_ALIGNED_PLUGIN="$code/muse_cmi_aligned_rewards.py"
export MUSE_MUCODEC_API_BASE=http://127.0.0.1:49321
export MUSECRITIC_RM_AUDIO_DIR="$output/decoded" MUSE_ROLLOUT_RESULT_PATH="$output/rollouts.jsonl"
export MUSE_GRPO_MAX_TOTAL_TOKENS=2048 MUSE_DECODE_DURATION=10.24 MUSE_DECODE_STEPS=20
export MUSE_MUCODEC_MAX_WORKERS=1 MUSECRITIC_RM_STRICT=true
export MUSE_INVALID_OUTPUT_LOG="$output/invalid_outputs.jsonl" MUSE_MIN_AUDIO_SECONDS=1
export REGIME_SCORER=http://127.0.0.1:49322 REGIME_BINDINGS="$output/reward-bindings.jsonl"
"$python" "$code/check_muse_base.py" --model "$base/models/Muse-0.6b" --output "$output/base-identity.json"
"$python" "$code/formal_inputs.py" --train "$study/data/muse-cmi-triple-train.jsonl" \
  --valid "$study/data/muse-cmi-triple-valid.jsonl" --manifest "$output/experiment.json" --reward "$REGIME_REWARD" --steps "$steps" --lr "$lr"
pids=()
cleanup(){ for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; wait 2>/dev/null || true; }
trap cleanup EXIT
metrics=/home/mengh/research/WildSongBench-eval/wsb_metrics
scorer_mode=()
if [[ "$REGIME_REWARD" == mean7 ]]; then scorer_mode=(--mean7-only); fi
"$metrics/.venvs/metrics/bin/python" -u "$code/reward_server.py" --bundle "$metrics" \
  --deps /home/mengh/research/reward-regimes/deps \
  --separator /home/mengh/.cache/torch/hub/checkpoints/955717e8-8726e21a.th \
  --receipt-dir "$output/scorer" --port 49322 "${scorer_mode[@]}" > "$output/scorer.log" 2>&1 &
pids+=("$!")
/home/mengh/miniconda3/envs/mucodec-cu128/bin/python -u "$code/mucodec_fixed_seed.py" \
  --source "$native/src/MuCodec/deploy/mucodec_serve.py" --model "$native/src/MuCodec/weights/mucodec.pt" \
  --host 127.0.0.1 --port 49321 > "$output/codec.log" 2>&1 &
pids+=("$!")
for port in 49321 49322; do
  ready=false
  for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then ready=true; break; fi
    for pid in "${pids[@]}"; do kill -0 "$pid" || exit 1; done
    sleep 5
  done
  [[ "$ready" == true ]] || exit 1
done
"$train/swift" rlhf --rlhf_type grpo --model_type qwen3 --model "$base/models/Muse-0.6b" \
  --reward_funcs regime_rm --external_plugins "$code/muse_plugin.py" \
  --tuner_type lora --lora_rank 8 --lora_alpha 16 --lora_dropout 0 --target_modules all-linear --seed 42 \
  --dataset "$study/data/muse-cmi-triple-train.jsonl" --use_vllm true --vllm_mode colocate \
  --vllm_gpu_memory_utilization .25 --vllm_max_num_seqs 2 --vllm_enforce_eager true \
  --vllm_engine_kwargs '{"kv_cache_memory_bytes":1073741824}' \
  --vllm_max_model_len 2048 --max_completion_length 512 --max_length 2048 \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 1 --num_generations 2 \
  --gradient_checkpointing true --max_steps "$steps" --learning_rate "$lr" --lr_scheduler_type constant \
  --warmup_ratio 0 --beta .05 --save_steps 10 --logging_steps 1 \
  --temperature 1 --repetition_penalty 1 --top_p 1 --top_k -1 --report_to none \
  --output_dir "$output/checkpoint" > "$output/train.log" 2>&1
if [[ "$steps" != 270 ]]; then
  printf '{"acceptance_only":true,"steps":%s,"formal_training":false}\n' "$steps" > "$output/acceptance-complete.json"
  exit 0
fi
adapter=$(find "$output/checkpoint" -path '*/checkpoint-270/adapter_model.safetensors')
[[ -n "$adapter" && "$adapter" != *$'\n'* ]] || exit 1
"$python" -u "$code/validate_muse.py" --model "$base/models/Muse-0.6b" \
  --valid "$study/data/muse-cmi-triple-valid.jsonl" --output "$output" \
  --baseline "$study/outputs/shared-muse-baseline" --adapter "$(dirname "$adapter")" \
  --steps 0,50,100,270 \
  --scorer "$REGIME_SCORER" --codec "$MUSE_MUCODEC_API_BASE" --reward "$REGIME_REWARD" \
  > "$output/validation.log" 2>&1
printf '{"training_steps":270,"full_valid_conditions":30,"test_run":false}\n' > "$output/completion.json"
