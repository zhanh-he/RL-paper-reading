#!/usr/bin/env bash
# A finite dependency queue, not a recurring task. Failure never starts a successor.
set -euo pipefail
study=/storage/zhanh_storage/songbench-priority-20261020
predecessor="$study/outputs/muse-richness-lr3e4-beta5e2-seed42/completion.json"
: "${MUSE_PREDECESSOR_PID:?}"
deadline=$((SECONDS+43200))
while [[ ! -f "$predecessor" ]]; do
  if ! kill -0 "$MUSE_PREDECESSOR_PID" 2>/dev/null; then
    echo 'Muse stopped without complete training/validation; successor not started'; exit 1
  fi
  [[ "$SECONDS" -lt "$deadline" ]] || { echo 'Finite queue timeout; no job interrupted'; exit 1; }
  sleep 30
done
while kill -0 "$MUSE_PREDECESSOR_PID" 2>/dev/null; do
  state=$(ps -p "$MUSE_PREDECESSOR_PID" -o stat= || true)
  [[ "$state" == Z* ]] && break
  [[ "$SECONDS" -lt "$deadline" ]] || exit 1
  sleep 5
done
exec bash "$study/code/run_diff_mean7_lab.sh"
