#!/usr/bin/env bash
set -euo pipefail
: "${EVAL_WORK:?}" "${BASE_PYTHON:?}" "${PREPARE_METRICS:?}"
mkdir -p "$EVAL_WORK" "$EVAL_WORK/cache"
export UV_CACHE_DIR="$EVAL_WORK/cache/uv"
export XDG_CACHE_HOME="$EVAL_WORK/cache/xdg"
archive="$EVAL_WORK/wsb_metrics.tar.gz"
if [[ ! -f "$archive" ]]; then
  curl --fail --location --retry 5 --output "$archive.part" \
    'https://huggingface.co/datasets/m-a-p/WildSongBench/resolve/e361b85a8d7365d3079fa9647641a87ce3c8c5b5/evaluation/wsb_metrics.tar.gz'
  mv "$archive.part" "$archive"
fi
printf '%s  %s\n' bd550b36f1fc43cad3254a9897dd3a078e50c4eb9e8d4f295ac4b5e81d1fc4fd "$archive" | sha256sum -c -
if [[ ! -f "$EVAL_WORK/wsb_metrics/release.json" ]]; then
  tar -xzf "$archive" -C "$EVAL_WORK"
fi
cd "$EVAL_WORK/wsb_metrics"
if [[ -n "${EVAL_RUNTIME_WORK:-}" ]]; then
  for pair in '.venvs:evaluator-venvs' '.runtime:evaluator-runtime'; do
    name=${pair%%:*}; target="$EVAL_RUNTIME_WORK/${pair#*:}"
    mkdir -p "$target"
    if [[ ! -e "$name" && ! -L "$name" ]]; then ln -s "$target" "$name"; fi
  done
fi
prepared=0
for attempt in 1 2 3; do
  if "$BASE_PYTHON" wsb_eval.py prepare --metrics "$PREPARE_METRICS" --download huggingface --install; then
    prepared=1
    break
  fi
  echo "Frozen preparation failed on attempt $attempt; retain downloads and retry" >&2
  sleep 15
done
if [[ "$prepared" != 1 ]]; then exit 5; fi
if [[ "${PREPARE_GENERATION:-0}" == 1 ]]; then
  "$BASE_PYTHON" - "$EVAL_WORK/models/YuE2-Vae-legacy" <<'PY'
import sys
from huggingface_hub import snapshot_download, hf_hub_download
snapshot_download('m-a-p/YuE2-Vae-legacy', revision='b54118f0fc462f08999d1ec07e88817f4ee3f770', local_dir=sys.argv[1])
PY
fi
"$BASE_PYTHON" -m venv "$EVAL_WORK/extras"
"$EVAL_WORK/extras/bin/python" -m pip install --no-deps audiobox-aesthetics==0.0.4
"$EVAL_WORK/extras/bin/python" -m pip install rich==14.3.3 submitit==1.5.4
parent_site=$("$BASE_PYTHON" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
"$EVAL_WORK/extras/bin/python" - "$parent_site" <<'PY'
import pathlib, sys, sysconfig
# Nested venvs inherit the system interpreter, not the invoking training venv.
pathlib.Path(sysconfig.get_path('purelib'), 'parent-runtime.pth').write_text(sys.argv[1] + '\n')
PY
"$EVAL_WORK/extras/bin/python" -c 'import torch, librosa, hydra, omegaconf, muq; from audiobox_aesthetics.infer import initialize_predictor; print("Supplementary imports verified:", torch.__version__)'
"$BASE_PYTHON" - <<'PY'
from huggingface_hub import snapshot_download, hf_hub_download
snapshot_download('OpenMuQ/MuQ-large-msd-iter', revision='0562a57814f6f8bbd9fdea0a25921a2fce1a841a', allow_patterns=['config.json', '*.bin', '*.safetensors'])
snapshot_download('facebook/audiobox-aesthetics', revision='9b1dd8e5df9af7216e836a98974fe3b82c56ded6', allow_patterns=['config.json', '*.safetensors'])
for repo, revision in [('OpenMuQ/MuQ-large-msd-iter', '0562a57814f6f8bbd9fdea0a25921a2fce1a841a'),
                       ('facebook/audiobox-aesthetics', '9b1dd8e5df9af7216e836a98974fe3b82c56ded6')]:
    path = hf_hub_download(repo, 'config.json', revision='main')
    if revision not in path:
        raise ValueError('Supplementary model main revision changed: ' + repo)
PY
"$BASE_PYTHON" - "$EVAL_WORK/preparation-receipt.json" "$PREPARE_METRICS" <<'PY'
import json, pathlib, sys
pathlib.Path(sys.argv[1]).write_text(json.dumps({'status': 'prepared', 'metrics': sys.argv[2], 'new_training': False}, indent=2) + '\n')
PY
