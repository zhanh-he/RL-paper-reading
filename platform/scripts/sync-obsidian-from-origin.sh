#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 /path/to/obsidian-vault" >&2
  exit 2
fi

script_dir=$(cd "$(dirname "$0")" && pwd)
repo_dir=$(cd "$script_dir/../.." && pwd)
source_dir=$(mktemp -d "${TMPDIR:-/tmp}/literature-sync.XXXXXXXX")
trap 'rm -rf -- "$source_dir"' EXIT

# Fetch the published source without touching the branch or working tree.
GIT_TERMINAL_PROMPT=0 git -C "$repo_dir" -c http.lowSpeedLimit=1024 -c http.lowSpeedTime=30 \
  fetch --quiet --no-tags --no-write-fetch-head origin main:refs/remotes/origin/main
revision=$(git -C "$repo_dir" rev-parse refs/remotes/origin/main)
for catalog in rl amt; do
  mkdir "$source_dir/$catalog"
  git -C "$repo_dir" show "$revision:platform/data/$catalog/literature.csv" > "$source_dir/$catalog/literature.csv"
done

node "$script_dir/sync-obsidian.mjs" --vault "$1" --source-dir "$source_dir" --write
