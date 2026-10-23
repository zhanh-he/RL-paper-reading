#!/usr/bin/env bash
set -euo pipefail
: "${MUSE_EVAL_RUN:?}"
study=/storage/zhanh_storage/songbench-priority-20261020
code="$study/code"
base=/home/mengh/research/vocal2accomp-muse
native=/home/mengh/research/vocal2accomp-muse-online
metrics=/home/mengh/research/WildSongBench-eval/wsb_metrics
python=/home/mengh/miniconda3/envs/muse-grpo/bin/python
output="$MUSE_EVAL_RUN/full-cohorts"
mkdir -p "$output"
exec 9>"$study/muse-full-cohorts.lock";flock -n 9 || exit 1
adapter=$(find "$MUSE_EVAL_RUN/checkpoint" -path '*/checkpoint-270/adapter_model.safetensors')
[[ -n "$adapter" && "$adapter" != *$'\n'* ]] || exit 1
export CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1
"$metrics/.venvs/metrics/bin/python" "$code/reward_server.py" --bundle "$metrics" \
 --deps /home/mengh/research/reward-regimes/deps --mean7-only \
 --separator /home/mengh/.cache/torch/hub/checkpoints/955717e8-8726e21a.th --port 49322 --receipt-dir "$output/scorer" > "$output/scorer.log" 2>&1 &
scorer=$!
/home/mengh/miniconda3/envs/mucodec-cu128/bin/python "$code/mucodec_fixed_seed.py" \
 --source "$native/src/MuCodec/deploy/mucodec_serve.py" --model "$native/src/MuCodec/weights/mucodec.pt" \
 --host 127.0.0.1 --port 49321 > "$output/codec.log" 2>&1 &
codec=$!
cleanup(){ kill "$scorer" "$codec" 2>/dev/null || true;wait 2>/dev/null || true; }
trap cleanup EXIT
for port in 49321 49322;do
 ready=false
 for _ in $(seq 1 180);do
  if curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1;then ready=true;break;fi
  kill -0 "$scorer";kill -0 "$codec";sleep 5
 done
 [[ "$ready" == true ]] || exit 1
done
for split in train test;do
 data="$study/data/muse-cmi-triple-train.jsonl";extra=()
 if [[ "$split" == test ]];then data="$study/data/wsb-test-20261023/muse-wsb-test.jsonl";extra=(--dataset-manifest "$study/data/wsb-test-20261023/manifest.json");fi
 "$python" -u "$code/validate_muse.py" --model "$base/models/Muse-0.6b" \
  --valid "$data" --output "$output" --baseline "$study/outputs/shared-muse-baseline-$split" \
  --adapter "$(dirname "$adapter")" --steps 0,50,100,270 --split "$split" "${extra[@]}" \
  --reward mean7 --scorer http://127.0.0.1:49322 --codec http://127.0.0.1:49321 --max-model-len 8192 > "$output/$split.log" 2>&1
done
printf '{"train_conditions":270,"test_conditions":192,"test_replicates":2,"steps":[0,50,100,270]}\n' > "$output/completion.json"
