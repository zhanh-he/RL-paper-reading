#!/usr/bin/env bash
set -euo pipefail
root=/home/mengh/research/YuE2-posttrain
code=/home/mengh/research/reward-regimes/code
output=/storage/zhanh_storage/reward-regimes-20261014
first="$output/yue2-richness-lr3e4-seed42"
while [[ ! -f "$first/completion.json" ]]; do
  state=$(ps -p "${FIRST_PID:?}" -o stat= || true)
  [[ -n "$state" && "$state" != Z* ]] || { echo 'First arm failed; queue will not hide or bypass it'; exit 1; }
  sleep 10
done
for reward in instrument arrangement; do
  "$root/.venv/bin/python" -u "$code/yue2_shared_baseline.py" \
    --baseline-root "$first/valid_000000" \
    --model "$root/models/YuE2-3B" --vae "$root/models/YuE2-Vae" \
    --data "$root/examples/cmi-pref-triple-text-lyrics-9to1-v2.json" \
    --reward "$reward" --scorer http://127.0.0.1:43140 \
    --output "$output/yue2-${reward}-lr3e4-seed42" --lr 3e-4 --steps 270
done
"$root/.venv/bin/python" -u "$code/render_yue2_replays.py" \
  --model "$root/models/YuE2-3B" --vae "$root/models/YuE2-Vae" \
  --data "$root/examples/cmi-pref-triple-text-lyrics-9to1-v2.json" --root "$output"
