#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/storage/zhanh_storage/open-music-rl-20261014}
MODEL=${1:?model required}
export HF_HOME="$ROOT/hf-cache" PIP_CACHE_DIR="$ROOT/pip-cache" PYTHONPYCACHEPREFIX="$ROOT/pycache"
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export NLTK_DATA="$ROOT/nltk_data"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"
PY="$ROOT/envs/$MODEL/bin/python"
"$PY" "$ROOT/code/test_open_head_policy.py"
extra=()
if [[ "$MODEL" == songbloom ]]; then extra+=(--reference-manifest "$ROOT/references.json"); fi
rewards=(richness instrument arrangement)
if [[ -n ${2:-} ]]; then rewards=("$2"); fi
for reward in "${rewards[@]}"; do
  "$PY" -u "$ROOT/code/train_open_head_grpo.py" --model "$MODEL" --assets "$ROOT/assets" \
    --sources "$ROOT/sources" --data "$ROOT/data.json" --examples "$ROOT/display-examples.json" \
    --output "$ROOT/outputs/$MODEL-$reward-3e-4" --reward "$reward" \
    --lr 3e-4 --steps 270 --seconds 60 --beta .01 "${extra[@]}" \
    --scorer "${SCORER_URL:-http://127.0.0.1:43140}" \
    >"$ROOT/logs/$MODEL-$reward.log" 2>&1
done
printf 'All three formal arms finished: %s\n' "$MODEL"
