#!/usr/bin/env bash
set -euo pipefail

root=/home/mengh/research/YuE2-posttrain
first="$root/outputs/cmi_triple_9to1_songeval_lr2e-5_train_20261007_lab5090"
first_pid=3497791

verify_run() {
  local output=$1 lr=$2
  python3 "$root/examples/collect_formal_run.py" \
    --run-root "$output" --reward-backend songeval \
    --learning-rate "$lr" --compute-dtype bf16 \
    --output "$output/verified_complete.json"
  python3 - "$output/verified_complete.json" <<'PY'
import json
import sys
assert json.load(open(sys.argv[1]))['status'] == 'verified-complete'
PY
}

while ps -p "$first_pid" -o args= 2>/dev/null | grep -Fq 'yue2_songeval_formal_9to1.py'; do
  sleep 60
done

verify_run "$first" 2e-5

wait_for_idle_gpu() {
  local apps
  while :; do
    if ! apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader); then
      sleep 60
      continue
    fi
    if [[ -z "$apps" ]]; then
      sleep 30
      if apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader) && [[ -z "$apps" ]]; then
        return
      fi
    fi
    sleep 60
  done
}

for lr in 1e-4 3e-4; do
  wait_for_idle_gpu
  output="$root/outputs/cmi_triple_9to1_songeval_lr${lr}_train_20261007_lab5090"
  if [[ -e "$output/experiment.json" ]]; then
    echo "Existing output requires manual review: $output" >&2
    exit 1
  fi
  echo "$(date -Is) starting formal SongEval LR=$lr" >&2
  LR="$lr" bash "$root/examples/run_lab5090_cmi_triple_9to1_songeval.sh" >"$output.log" 2>&1
  verify_run "$output" "$lr"
  echo "$(date -Is) completed formal SongEval LR=$lr" >&2
done
