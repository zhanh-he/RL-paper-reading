#!/usr/bin/env bash
# Finite job dependencies, not a recurring monitor or scheduled automation.
set -euo pipefail
study=/storage/zhanh_storage/songbench-priority-20261020
exec 9>"$study/progression-20261023.lock"
flock -n 9 || exit 1
current="$study/outputs/muse-mean7-lr3e4-beta5e2-seed42-v3"
exec 8>"$current/training.lock"
flock 8
[[ -f "$current/completion.json" ]] || { echo 'Muse Mean7 failed; dependency queue stops'; exit 1; }
flock -u 8
# Do not launch into another workload after the dependency releases its GPU.
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
[[ "$used" =~ ^[0-9]+$ && "$used" -lt 2048 ]] || { echo 'Other GPU workload present; no interruption'; exit 1; }
DIFF_LR=1e-4 DIFF_OUTPUT="$study/outputs/diff-mean7-lr1e4-formal-20261023" \
  bash "$study/code/run_diff_mean7_lab.sh"
REGIME_REWARD=richness MUSE_OUTPUT="$study/outputs/muse-richness-lr3e4-beta5e2-seed42-v2" \
  bash "$study/code/run_muse_lab_priority.sh"
for run in "$current" "$study/outputs/muse-richness-lr3e4-beta5e2-seed42-v2";do
 MUSE_EVAL_RUN="$run" bash "$study/code/eval_muse_cohorts_lab.sh"
done
printf 'Finite training dependencies complete. No repeating follow-up is installed.\n'
