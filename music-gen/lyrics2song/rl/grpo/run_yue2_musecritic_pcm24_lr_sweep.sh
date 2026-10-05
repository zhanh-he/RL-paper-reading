#!/usr/bin/env bash
set -euo pipefail

root=/home/mengh/research/YuE2-posttrain
python="$root/.venv/bin/python"
script="$root/examples/yue2_songeval_longrun.py"
queue="$root/outputs/musecritic_pcm24_lr_sweep_20261005"
mkdir -p "$queue"
exec 9>"$queue/runner.lock"
flock -n 9 || { echo 'Another MuseCritic LR sweep is active'; exit 1; }
export CUDA_VISIBLE_DEVICES=0

log() { echo "$(date -Is) $*" | tee -a "$queue/status.log"; }

wait_for_idle_gpu() {
  while :; do
    if [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')" ]]; then
      sleep 30
      if [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')" ]]; then
        return
      fi
    fi
    log 'waiting for idle GPU'
    sleep 60
  done
}

run_arm() {
  local tag="$1" lr="$2" output="$root/outputs/musecritic_grpo_pcm24_${tag}_20261005"
  mkdir -p "$output"
  local common=(
    --model "$root/models/YuE2-3B"
    --vae "$root/models/YuE2-Vae"
    --start-adapter "$root/outputs/songeval_grpo_pilot_20260929/adapter"
    --reward-backend musecritic
    --musecritic-python /home/mengh/miniconda3/envs/musecritic/bin/python
    --musecritic-repo /home/mengh/research/vocal2accomp-muse/src/MuseCritic
    --musecritic-model /home/mengh/research/musecritic_beat_eval/models/MuseCritic
    --musecritic-max-new-tokens 512
    --output "$output"
    --learning-rate "$lr"
    --max-tokens 600
    --checkpoint-every 25
  )
  log "starting $tag smoke"
  if ! "$python" -u "$script" "${common[@]}" --max-steps 2 >"$output/smoke.log" 2>&1; then
    log "$tag smoke failed; inspect $output/smoke.log"
    return 1
  fi
  log "starting $tag to 100 updates"
  if ! "$python" -u "$script" "${common[@]}" --max-steps 100 >"$output/run.log" 2>&1; then
    log "$tag training failed; inspect $output/run.log"
    return 1
  fi
  for step in 25 50; do
    log "evaluating $tag step $step"
    if ! "$python" -u "$script" "${common[@]}" --max-steps 100 --eval-step "$step" \
      >"$output/eval_step_${step}.log" 2>&1; then
      log "$tag evaluation failed at step $step"
      return 1
    fi
  done
  log "completed $tag with 0/1/5/25/50/100 held-out receipts"
}

for spec in 'lr1e4 1e-4' 'lr1e3 1e-3' 'lr1e2 1e-2'; do
  read -r tag lr <<<"$spec"
  wait_for_idle_gpu
  run_arm "$tag" "$lr"
done

wait_for_idle_gpu
kl_output="$root/outputs/reference_kl_musecritic_pcm24_sweep_20261005"
mkdir -p "$kl_output"
cp "$root/outputs/reference_kl_20260930/reference_tokens.json" "$kl_output/reference_tokens.json"
log 'starting fixed-reference KL audit'
"$python" -u "$root/examples/probe_yue2_reference_kl.py" \
  --model "$root/models/YuE2-3B" \
  --vae "$root/models/YuE2-Vae" \
  --reference-adapter "$root/outputs/songeval_grpo_pilot_20260929/adapter" \
  --arm 2e-5 "$root/outputs/musecritic_grpo_pcm24_lr2e5_20261001" \
  --arm 1e-4 "$root/outputs/musecritic_grpo_pcm24_lr1e4_20261005" \
  --arm 1e-3 "$root/outputs/musecritic_grpo_pcm24_lr1e3_20261005" \
  --arm 1e-2 "$root/outputs/musecritic_grpo_pcm24_lr1e2_20261005" \
  --steps 5 25 50 100 \
  --max-tokens 600 \
  --output "$kl_output" >"$kl_output/run.log" 2>&1
log 'MuseCritic PCM24 LR sweep and KL audit finished'
