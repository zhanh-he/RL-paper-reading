"""Publish aggregate GRPO receipts without private song IDs or checkpoint weights."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="ARM=RECEIPT_JSON")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"status": "measured_segment_event_head_grpo_ablation",
              "evaluation": "5.12-second held-out YouChorale-Pro segments, 50 ms onset; offset max(50 ms, 20% reference duration)",
              "baseline": None, "arms": {}}
    common = None
    for item in args.run:
        arm, sep, path = item.partition("=")
        if not sep or arm in result["arms"]:
            raise ValueError("Pass each arm once as ARM=RECEIPT_JSON")
        receipt = json.loads(Path(path).read_text())
        if receipt["status"] != "measured_event_head_grpo_segment_pilot" or receipt["reward_arm"] != arm:
            raise ValueError(f"Not a matching event GRPO receipt: {path}")
        identity = (receipt["checkpoint_sha256"], receipt["seed"],
                    receipt["train_original_songs"], receipt["test_original_songs"],
                    receipt["segment_index"], receipt["before"])
        if common is not None and identity != common:
            raise ValueError("Arms must share the same checkpoint, seed, splits and baseline")
        common = identity
        result["baseline"] = receipt["before"]
        result["protocol"] = {
            "checkpoint_sha256": receipt["checkpoint_sha256"],
            "seed": receipt["seed"],
            "train_original_songs": receipt["train_original_songs"],
            "test_original_songs": receipt["test_original_songs"],
            "segment_index": receipt["segment_index"],
            "segment_seconds": receipt["segment_seconds"],
            "group_size": receipt["group_size"],
            "max_tokens": receipt["max_tokens"],
            "update_epochs": receipt["update_epochs"],
            "learning_rate": receipt["lr"],
            "kl_beta": receipt["kl_beta"],
            "start_sigma": receipt["start_sigma"],
            "duration_sigma": receipt["duration_sigma"],
        }
        result["arms"][arm] = {
            "steps_requested": receipt["steps_requested"],
            "steps_updated": receipt["steps_updated"],
            "milestones": {step: {
                "optimizer_step": stage["optimizer_step"],
                "updated_steps": stage["updated_steps"],
                "metrics": stage["metrics"],
            } for step, stage in receipt["milestones"].items()},
            "after": receipt["after"],
        }
    serialized = json.dumps(result, indent=2) + "\n"
    if any(key in serialized for key in ("per_segment", "train_selected", "test_selected")):
        raise ValueError("Private per-song data leaked into public receipt")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized)
    print(f"Wrote public aggregate for {', '.join(result['arms'])}")


if __name__ == "__main__":
    main()
