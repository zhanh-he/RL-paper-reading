#!/usr/bin/env bash
set -u

root=/home/mengh/research/YuE2-posttrain
python="$root/.venv/bin/python"
script="$root/examples/yue2_songeval_longrun.py"
scorer=/home/mengh/miniconda3/envs/pytorch_env/bin/python
songeval=/home/mengh/research/SongEval-audit
source_adapter="$root/outputs/songeval_grpo_pilot_20260929/adapter"
queue="$root/outputs/songeval_lr_sweep_20260930"
mkdir -p "$queue"
exec 9>"$queue/runner.lock"
flock -n 9 || { echo 'Another LR sweep runner is active'; exit 1; }

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

run_arm() {
  local name="$1" lr="$2" output="$root/outputs/$1"
  mkdir -p "$output"
  echo "$(date -Is) starting $name lr=$lr" | tee -a "$queue/status.log"
  "$python" -u "$script" \
    --model "$root/models/YuE2-3B" \
    --vae "$root/models/YuE2-Vae" \
    --start-adapter "$source_adapter" \
    --songeval "$songeval" \
    --scorer-python "$scorer" \
    --output "$output" \
    --learning-rate "$lr" \
    --max-steps 100 \
    --max-tokens 600 \
    --checkpoint-every 25 >"$output/run.log" 2>&1
  local result=$?
  echo "$(date -Is) training $name exit=$result" | tee -a "$queue/status.log"
  if (( result != 0 )); then
    return
  fi
  for step in 25 50; do
    "$python" -u "$script" \
      --model "$root/models/YuE2-3B" \
      --vae "$root/models/YuE2-Vae" \
      --start-adapter "$source_adapter" \
      --songeval "$songeval" \
      --scorer-python "$scorer" \
      --output "$output" \
      --learning-rate "$lr" \
      --max-steps 100 \
      --max-tokens 600 \
      --eval-step "$step" >"$output/eval_step_${step}.log" 2>&1
    echo "$(date -Is) evaluation $name step=$step exit=$?" | tee -a "$queue/status.log"
  done
}

export CUDA_VISIBLE_DEVICES=0
wait_for_idle_gpu
run_arm songeval_grpo_lr1e3_20260930 1e-3
wait_for_idle_gpu
run_arm songeval_grpo_lr1e2_20260930 1e-2
echo "$(date -Is) LR sweep finished" | tee -a "$queue/status.log"
