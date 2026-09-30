#!/usr/bin/env bash
set -u

root=/home/mengh/research/YuE2-posttrain
python="$root/.venv/bin/python"
source_adapter="$root/outputs/songeval_grpo_pilot_20260929/adapter"
sweep="$root/outputs/songeval_lr_sweep_20260930"
queue="$root/outputs/yue2_post_sweep_20260930"
kl_output="$root/outputs/reference_kl_20260930"
muse_output="$root/outputs/musecritic_grpo_lr2e5_20260930"
mkdir -p "$queue" "$kl_output" "$muse_output"
exec 9>"$queue/runner.lock"
flock -n 9 || { echo 'Another post-sweep runner is active'; exit 1; }
export CUDA_VISIBLE_DEVICES=0

wait_for_sweep() {
  while ! grep -q 'LR sweep finished' "$sweep/status.log" 2>/dev/null; do
    echo "$(date -Is) waiting for LR sweep" >>"$queue/status.log"
    sleep 120
  done
}

wait_for_idle_gpu() {
  while :; do
    if [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')" ]]; then
      sleep 30
      if [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')" ]]; then
        return
      fi
    fi
    echo "$(date -Is) waiting for idle GPU" >>"$queue/status.log"
    sleep 60
  done
}

log() {
  echo "$(date -Is) $*" | tee -a "$queue/status.log"
}

wait_for_sweep
wait_for_idle_gpu
log 'starting offline reference KL probe'
"$python" -u "$root/examples/probe_yue2_reference_kl.py" \
  --model "$root/models/YuE2-3B" \
  --vae "$root/models/YuE2-Vae" \
  --reference-adapter "$source_adapter" \
  --arm 2e-5 "$root/outputs/songeval_grpo_longrun_20260930" \
  --arm 1e-4 "$root/outputs/songeval_grpo_lr1e4_retry_20260930" \
  --steps 5 25 50 100 \
  --max-tokens 600 \
  --output "$kl_output" >"$kl_output/run.log" 2>&1
kl_result=$?
log "offline reference KL probe exit=$kl_result"

wait_for_idle_gpu
common=(
  --model "$root/models/YuE2-3B"
  --vae "$root/models/YuE2-Vae"
  --start-adapter "$source_adapter"
  --reward-backend musecritic
  --musecritic-python /home/mengh/miniconda3/envs/musecritic/bin/python
  --musecritic-repo /home/mengh/research/vocal2accomp-muse/src/MuseCritic
  --musecritic-model /home/mengh/research/musecritic_beat_eval/models/MuseCritic
  --musecritic-max-new-tokens 512
  --output "$muse_output"
  --learning-rate 2e-5
  --max-tokens 600
  --checkpoint-every 25
)
log 'starting MuseCritic 2-step smoke test'
"$python" -u "$root/examples/yue2_songeval_longrun.py" \
  "${common[@]}" --max-steps 2 >"$muse_output/smoke.log" 2>&1
smoke_result=$?
log "MuseCritic smoke exit=$smoke_result"
if (( smoke_result != 0 )); then
  log 'MuseCritic full run withheld because smoke failed'
  exit "$smoke_result"
fi

log 'resuming MuseCritic to 100 optimizer steps'
"$python" -u "$root/examples/yue2_songeval_longrun.py" \
  "${common[@]}" --max-steps 100 >"$muse_output/run.log" 2>&1
train_result=$?
log "MuseCritic 100-step training exit=$train_result"
if (( train_result != 0 )); then
  exit "$train_result"
fi

for step in 25 50; do
  "$python" -u "$root/examples/yue2_songeval_longrun.py" \
    "${common[@]}" --max-steps 100 --eval-step "$step" \
    >"$muse_output/eval_step_${step}.log" 2>&1
  log "MuseCritic heldout evaluation step=$step exit=$?"
done
log 'post-sweep queue finished'
