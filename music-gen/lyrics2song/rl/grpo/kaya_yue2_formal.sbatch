#!/usr/bin/env bash
#SBATCH --job-name=yue2-cmi-9to1
#SBATCH --account=ems011
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-32gb:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=36:00:00
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
  smoke) steps=2; output="$root/outputs/cmi_triple_9to1_songeval_lr1e4_smoke_20261006"; extra=(--smoke) ;;
  train) steps=270; output="$root/outputs/cmi_triple_9to1_songeval_lr1e4_train_20261006"; extra=() ;;
  *) printf 'Unknown FORMAL_MODE=%s\n' "$mode" >&2; exit 2 ;;
esac

cd "$root/YuE2-posttrain"
exec "$root/.venv/bin/python" -u examples/yue2_songeval_longrun.py \
  --model "$root/models/YuE2-3B" \
  --vae "$root/models/YuE2-Vae" \
  --songeval "$root/SongEval" \
  --scorer-python "$root/.venv/bin/python" \
  --dataset-manifest "$root/YuE2-posttrain/examples/cmi-pref-triple-text-lyrics-9to1-v2.json" \
  --expected-dataset-sha256 c39d43e81793bee3f312c7028a0483b5da41068db4544acd258115043101e46f \
  --reward-backend songeval \
  --output "$output" \
  --max-steps "$steps" \
  --max-tokens 600 \
  --learning-rate 1e-4 \
  --kl-beta 0.01 \
  --compute-dtype fp16 \
  --checkpoint-every 25 \
  "${extra[@]}"
