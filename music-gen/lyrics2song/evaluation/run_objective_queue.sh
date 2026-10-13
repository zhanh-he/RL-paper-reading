#!/usr/bin/env bash
set -euo pipefail
root=${YUE2_ROOT:-/home/mengh/research/YuE2-posttrain}
work=${EVAL_WORK:-/home/mengh/research/WildSongBench-eval}
storage=${EVAL_STORAGE:-/storage/zhanh_storage/lyrics-objective-eval-20261013}
code="$work/code"
python="$root/.venv/bin/python"
export CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false
export HF_HOME=/home/mengh/.cache/huggingface
export OMP_NUM_THREADS=8
mkdir -p "$storage/runs"
exec 9>"$storage/queue.lock"
flock -n 9 || { echo 'Evaluation queue already running'; exit 2; }

gpu_available() {
  local pids
  pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
  if [[ -n "$pids" ]]; then
    echo "GPU occupied by another job: $pids. Evaluation stops without interrupting it." >&2
    exit 3
  fi
}

# Setup is CPU/download-only; generation can start before the scorer install.
# Interleave arms, so matched before/after evidence arrives before one whole
# baseline campaign monopolizes the GPU for every dataset.
for ((start=0; start<192; start+=4)); do
  for dataset in wsb cmi; do
    count=192; source="$code/reproduction_manifest.jsonl"; split=()
    if [[ "$dataset" == cmi ]]; then
      count=121; source="$code/cmi_test.jsonl"; split=(--split "$code/cmi-pref-triple-9to1-split.json")
    fi
    if ((start>=count)); then continue; fi
    end=$((start+4)); ((end>count)) && end=$count
    for arm in baseline songeval-2e-5 musecritic-2e-5 songeval-1e-4 musecritic-1e-4 songeval-3e-4 musecritic-3e-4; do
      adapter=()
      if [[ "$arm" != baseline ]]; then
        reward=${arm%%-*}; lr=${arm#*-}
        if [[ "$reward" == songeval ]]; then
          run="$root/outputs/cmi_triple_9to1_${reward}_lr${lr}_train_20261007_lab5090"
        else
          run="$root/outputs/cmi_triple_9to1_${reward}_lr${lr}_train_20261006"
        fi
        adapter=(--adapter "$run/checkpoints/step_000270/adapter")
      fi
      output="$storage/runs/yue2-${arm}/$dataset"
      gpu_available
      echo "$(date -Iseconds) GENERATE YuE2-$arm $dataset $start:$end"
      "$python" -u "$code/generate_objective.py" --dataset "$dataset" --source "$source" "${split[@]}" \
        --model "$root/models/YuE2-3B" --vae "$storage/models/YuE2-Vae-legacy" \
        --arm "YuE2-$arm" "${adapter[@]}" --output "$output" --start "$start" --end "$end"
      while [[ ! -f "$work/wsb_metrics/validation/preparation.json" ]]; do
        if ! kill -0 "${PREPARE_PID:?Scorer prepare PID must be supplied}" 2>/dev/null; then
          echo 'Scorer preparation failed; inspect prepare.log' >&2; exit 4
        fi
        sleep 15
      done
      gpu_available
      echo "$(date -Iseconds) SCORE YuE2-$arm $dataset $start:$end"
      "$python" -u "$code/score_objective.py" --bundle "$work/wsb_metrics" --run "$output" \
        --source "$source" "${split[@]}" --start "$start" --end "$end" \
        --extras-python "$work/extras/bin/python" --songeval /home/mengh/research/SongEval-audit
    done
  done
done
echo "$(date -Iseconds) All available YuE2 final-step test arms complete"
