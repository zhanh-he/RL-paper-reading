#!/usr/bin/env bash
set -u

root=/home/mengh/research/YuE2-posttrain
output="$root/outputs/musecritic_grpo_pcm24_lr2e5_20261001"
mkdir -p "$output"
exec 9>"$output/runner.lock"
flock -n 9 || { echo 'Another canonical MuseCritic runner is active'; exit 1; }
export CUDA_VISIBLE_DEVICES=0

while :; do
  if [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')" ]]; then
    sleep 30
    if [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')" ]]; then
      break
    fi
  fi
  echo "$(date -Is) waiting for idle GPU" >>"$output/status.log"
  sleep 60
done

common=(
  --model "$root/models/YuE2-3B"
  --vae "$root/models/YuE2-Vae"
  --start-adapter "$root/outputs/songeval_grpo_pilot_20260929/adapter"
  --reward-backend musecritic
  --musecritic-python /home/mengh/miniconda3/envs/musecritic/bin/python
  --musecritic-repo /home/mengh/research/vocal2accomp-muse/src/MuseCritic
  --musecritic-model /home/mengh/research/musecritic_beat_eval/models/MuseCritic
  --musecritic-max-new-tokens 512
  --output "$output"
  --learning-rate 2e-5
  --max-tokens 600
  --checkpoint-every 25
)

echo "$(date -Is) starting PCM24 FLAC MuseCritic smoke" >>"$output/status.log"
"$root/.venv/bin/python" -u "$root/examples/yue2_songeval_longrun.py" \
  "${common[@]}" --max-steps 2 >"$output/smoke.log" 2>&1
result=$?
echo "$(date -Is) smoke exit=$result" >>"$output/status.log"
if (( result != 0 )); then
  exit "$result"
fi

echo "$(date -Is) resuming PCM24 FLAC MuseCritic to step 100" >>"$output/status.log"
"$root/.venv/bin/python" -u "$root/examples/yue2_songeval_longrun.py" \
  "${common[@]}" --max-steps 100 >"$output/run.log" 2>&1
result=$?
echo "$(date -Is) step-100 training exit=$result" >>"$output/status.log"
if (( result != 0 )); then
  exit "$result"
fi

for step in 25 50; do
  "$root/.venv/bin/python" -u "$root/examples/yue2_songeval_longrun.py" \
    "${common[@]}" --max-steps 100 --eval-step "$step" \
    >"$output/eval_step_${step}.log" 2>&1
  echo "$(date -Is) heldout step=$step exit=$?" >>"$output/status.log"
done
echo "$(date -Is) PCM24 FLAC MuseCritic run finished" >>"$output/status.log"
