"""Capture selected online Beat-v2 rollout WAVs before the trainer overwrites them."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--from-step", type=int, required=True)
    parser.add_argument("--until-step", type=int, default=100)
    parser.add_argument("--min-reward", type=float, default=0.25)
    parser.add_argument("--max-coverage", type=float, default=0.20)
    parser.add_argument("--max-captures", type=int, default=5)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    candidate_path = run_dir / "beat_v2_candidate.wav"
    metrics_path = run_dir / "metrics.jsonl"
    output_dir = run_dir / "audit_candidates"
    output_dir.mkdir(exist_ok=True)
    seen_step = args.from_step
    captures = 0

    while seen_step < args.until_step and captures < args.max_captures:
        if metrics_path.exists():
            for line in metrics_path.read_text().splitlines():
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                step = int(item["step"])
                if step <= seen_step:
                    continue
                seen_step = step
                if len(item["scores"]) != 2:
                    continue
                score = item["scores"][1]
                if score["reward"] < args.min_reward or score["rms_coverage"] > args.max_coverage:
                    continue
                if not candidate_path.exists():
                    continue
                output = output_dir / f"step_{step:04d}_candidate_2.wav"
                shutil.copy2(candidate_path, output)
                receipt = {
                    "step": step,
                    "candidate_index": 2,
                    "training_seconds": 6,
                    "scores": score,
                    "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                    "source": "online Beat-v2 rollout; not a fixed-seed checkpoint replay",
                }
                output.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
                captures += 1
                print(json.dumps({"captured": str(output), "reward": score["reward"], "coverage": score["rms_coverage"]}), flush=True)
                if captures >= args.max_captures:
                    break
        time.sleep(0.25)


if __name__ == "__main__":
    main()
