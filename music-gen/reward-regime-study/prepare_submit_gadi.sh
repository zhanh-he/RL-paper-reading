#!/usr/bin/env bash
# One finite preparation/submission process, not a monitor or scheduled task.
set -euo pipefail
ROOT=/scratch/iv96/hm6416/open-music-rl-20261014
base=/g/data/wa66/hanyu/YuE2-posttrain/.venv/bin/python
metrics=/scratch/iv96/hm6416/wsb-objective-20261013/wsb_metrics/.venvs/metrics/bin/python
runtime=/scratch/iv96/hm6416/wsb-objective-20261013/container-runtime.json
export HF_HOME="$ROOT/hf-cache" PIP_CACHE_DIR="$ROOT/pip-cache" OMP_NUM_THREADS=2
MODEL=${1:?}; PROJECT=${2:?}
cd "$ROOT"
"$base" "$ROOT/code/prepare_open_assets.py" --root "$ROOT/assets" --model "$MODEL"
"$base" "$ROOT/code/container_command.py" "$runtime" "$metrics" \
  "$ROOT/code/prepare_open_env.py" --root "$ROOT" --model "$MODEL"
"$base" "$ROOT/code/container_command.py" "$runtime" "$ROOT/envs/$MODEL/bin/python" \
  "$ROOT/code/test_open_head_policy.py"
if [[ "$MODEL" == songbloom ]]; then
  "$base" "$ROOT/code/prepare_cmi_references.py" --root "$ROOT"
fi
for reward in richness instrument arrangement; do
  qsub -P "$PROJECT" -p 1023 -N "rr-$MODEL-${reward:0:1}" \
    -v "MODEL=$MODEL,REWARD=$reward" -o "$ROOT/logs/pbs-$MODEL-$reward.out" \
    -e "$ROOT/logs/pbs-$MODEL-$reward.err" "$ROOT/code/gadi_open_formal.pbs" \
    >> "$ROOT/logs/submitted-$MODEL.txt"
done
