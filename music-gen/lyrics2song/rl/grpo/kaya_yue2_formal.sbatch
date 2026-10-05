#!/usr/bin/env bash
#SBATCH --job-name=yue2-cmi-formal
#SBATCH --account=ems011
#SBATCH --partition=ondemand-gpu
#SBATCH --gres=gpu:v100-32gb:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --nice=10000
#SBATCH --output=/scratch/ems011/zhe/yue2-formal/logs/%x-%j.out
#SBATCH --error=/scratch/ems011/zhe/yue2-formal/logs/%x-%j.err
set -euo pipefail

root=/scratch/ems011/zhe/yue2-formal
export HF_HOME="$root/hf-cache"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
mode=${FORMAL_MODE:?Set FORMAL_MODE=smoke or train}
case "$mode" in
  smoke) steps=2; output="$root/outputs/cmi_pref_songeval_lr1e4_smoke_20261005"; extra=(--smoke) ;;
  train) steps=100; output="$root/outputs/cmi_pref_songeval_lr1e4_formal_20261005"; extra=() ;;
  *) printf 'Unknown FORMAL_MODE=%s\n' "$mode" >&2; exit 2 ;;
esac

cd "$root/YuE2-posttrain"
exec "$root/.venv/bin/python" -u examples/yue2_songeval_longrun.py \
  --model "$root/models/YuE2-3B" \
  --vae "$root/models/YuE2-Vae" \
  --songeval "$root/SongEval" \
  --scorer-python "$root/.venv/bin/python" \
  --dataset-manifest "$root/YuE2-posttrain/examples/formal-lyrics2song-v1.json" \
  --expected-dataset-sha256 c29217883289d3717c510d671927319d5e7acb70ad00443d2a3d9384653ac23b \
  --reward-backend songeval \
  --output "$output" \
  --max-steps "$steps" \
  --max-tokens 600 \
  --learning-rate 1e-4 \
  --kl-beta 0.01 \
  --compute-dtype fp16 \
  --checkpoint-every 25 \
  "${extra[@]}"
