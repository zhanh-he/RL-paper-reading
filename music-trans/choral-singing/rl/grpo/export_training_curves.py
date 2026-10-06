"""Export binned ChoralGRPO training traces without recording identifiers."""

import argparse
import json
import math
from pathlib import Path


ARMS = ("combined", "onset", "onset_offset", "frame", "coverage", "continuity")


def mean(values):
    return round(sum(values) / len(values), 6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bin-size", type=int, default=25)
    args = parser.parse_args()
    if args.bin_size <= 0 or 1000 % args.bin_size:
        parser.error("bin-size must evenly divide 1000")

    runs = {}
    for arm in ARMS:
        path = args.runs_root / f"event_{arm}_16songs_1000steps" / "trace.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != 1000 or any(row.get("step") != index for index, row in enumerate(rows, 1)):
            raise ValueError(f"Unexpected training step sequence: {arm}")
        points = []
        for start in range(0, len(rows), args.bin_size):
            active = [row for row in rows[start:start + args.bin_size] if not row["skipped"]]
            if active and any(not all(math.isfinite(row[key]) for key in ("reward_mean", "kl")) for row in active):
                raise ValueError(f"Non-finite training metric: {arm}, step {start + 1}")
            points.append({
                "step": start + args.bin_size,
                "reward": mean([row["reward_mean"] for row in active]) if active else None,
                "kl": mean([row["kl"] for row in active]) if active else None,
                "updates": len(active),
            })
        first = rows[0]
        runs[arm] = {
            "steps": len(rows),
            "updated_steps": sum(not row["skipped"] for row in rows),
            "first_step": {
                "step": 1,
                "reward": round(first["reward_mean"], 6) if not first["skipped"] else None,
                "kl": round(first["kl"], 6) if not first["skipped"] else None,
            },
            "points": points,
        }

    result = {
        "status": "measured_training_trace_window_means",
        "bin_size": args.bin_size,
        "scope": "training rollouts, not held-out evaluation",
        "reward_definition": "mean reward for non-skipped GRPO updates in each window",
        "kl_definition": "mean logged KL estimate for non-skipped GRPO updates in each window",
        "runs": runs,
    }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
