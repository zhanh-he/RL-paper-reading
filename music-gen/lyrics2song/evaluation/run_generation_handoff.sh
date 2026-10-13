#!/usr/bin/env bash
set -euo pipefail
: "${EVAL_STORAGE:?}" "${EVAL_WORK:?}"
# A stopped controller's current child retains the lock and finishes normally.
exec 8>"$EVAL_STORAGE/queue.lock"
flock 8
flock -u 8
exec 8>&-
exec bash "$EVAL_WORK/code/run_objective_queue.sh"
