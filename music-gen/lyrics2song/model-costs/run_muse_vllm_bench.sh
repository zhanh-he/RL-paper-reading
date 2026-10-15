#!/usr/bin/env bash
set -euo pipefail
study=/scratch/wa66/hm6416/model-costs-vllm-20261015
root=/g/data/wa66/hanyu/vocal2accomp-muse
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1
export COSTS_CODEC_SOURCE="$root/src/MuCodec/deploy/mucodec_serve.py"
export COSTS_CODEC_COUNT="$study/mucodec-parameters.json"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
LD_LIBRARY_PATH="/apps/python3/3.10.4/lib:${LD_LIBRARY_PATH:-}" \
 "$root/envs/mucodec-py310-cu128/bin/python" "$study/profile_codec_server.py" \
 --model "$root/src/MuCodec/weights/mucodec.pt" --host 127.0.0.1 --port 48891 > "$study/codec.log" 2>&1 &
codec=$!
trap 'kill "$codec" 2>/dev/null || true; wait "$codec" 2>/dev/null || true' EXIT
ready=false
for _ in $(seq 1 120); do
 if curl -fsS http://127.0.0.1:48891/health >/dev/null 2>&1; then ready=true; break; fi
 kill -0 "$codec" || exit 1
 sleep 3
done
[[ "$ready" == true ]] || exit 1
/g/data/wa66/hanyu/YuE2-posttrain/.venv/bin/python \
 /scratch/iv96/hm6416/open-music-rl-20261014/code/container_command.py \
 /scratch/iv96/hm6416/wsb-objective-20261013/container-runtime.json \
 /scratch/iv96/hm6416/wsb-objective-20261013/wsb_metrics/.venvs/metrics/bin/python -u \
 "$study/bench_muse_vllm.py" --model "$root/models/Muse-0.6b" \
 --valid "$root/experiments/cmi_aligned_lr_20261012/data/muse-cmi-triple-valid.jsonl" \
 --codec http://127.0.0.1:48891 --output "$study/muse-vllm-timing.json" > "$study/benchmark.log" 2>&1
