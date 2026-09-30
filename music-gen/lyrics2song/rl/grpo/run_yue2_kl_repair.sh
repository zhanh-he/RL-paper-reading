#!/usr/bin/env bash
set -u

root=/home/mengh/research/YuE2-posttrain
output="$root/outputs/reference_kl_20260930"
mkdir -p "$output"
exec 9>"$output/repair.lock"
flock -n 9 || { echo 'Another KL repair runner is active'; exit 1; }
export CUDA_VISIBLE_DEVICES=0

while pgrep -f "^bash $root/examples/run_yue2_post_sweep.sh$" >/dev/null; do
  echo "$(date -Is) waiting for MuseCritic runner" >>"$output/repair_status.log"
  sleep 120
done
while :; do
  if [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')" ]]; then
    sleep 30
    if [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')" ]]; then
      break
    fi
  fi
  echo "$(date -Is) waiting for idle GPU" >>"$output/repair_status.log"
  sleep 60
done

if [[ -e "$output/kl.json" ]]; then
  mv "$output/kl.json" "$output/kl_invalid_inf_mask.json"
fi
echo "$(date -Is) starting corrected KL probe" >>"$output/repair_status.log"
"$root/.venv/bin/python" -u "$root/examples/probe_yue2_reference_kl.py" \
  --model "$root/models/YuE2-3B" \
  --vae "$root/models/YuE2-Vae" \
  --reference-adapter "$root/outputs/songeval_grpo_pilot_20260929/adapter" \
  --arm 2e-5 "$root/outputs/songeval_grpo_longrun_20260930" \
  --arm 1e-4 "$root/outputs/songeval_grpo_lr1e4_retry_20260930" \
  --steps 5 25 50 100 \
  --max-tokens 600 \
  --output "$output" >"$output/repair.log" 2>&1
result=$?
echo "$(date -Is) corrected KL probe exit=$result" >>"$output/repair_status.log"
exit "$result"
