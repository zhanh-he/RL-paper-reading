"""Verify one formal YuE2 run and emit a public, lyrics-free curve receipt."""

import argparse
import hashlib
import json
import math
from pathlib import Path


DATASET_SHA256 = "c29217883289d3717c510d671927319d5e7acb70ad00443d2a3d9384653ac23b"
CHECKPOINTS = (0, 1, 25, 50, 100)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def checked_audio(root, relative, expected):
    path = root / relative
    if not path.is_file() or digest(path) != expected:
        raise ValueError(f"Missing or mismatched scored audio: {path}")


def collect(root, backend, learning_rate, compute_dtype="bf16"):
    config_path = root / "experiment.json"
    if not config_path.is_file():
        return {"status": "submitted-unverified", "train_curve": [], "validation": {}}
    config = json.loads(config_path.read_text())
    expected = {
        "dataset_sha256": DATASET_SHA256,
        "dataset_protocol": "cmi-pref-lyrics-no-ref-decontaminated-v1",
        "reward_backend": backend,
        "train_prompt_count": 234,
        "validation_prompt_count": 59,
        "kl_beta": 0.01,
        "compute_dtype": compute_dtype,
        "smoke_only": False,
        "max_tokens": 600,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Experiment config {key}={config.get(key)!r}, expected {value!r}")
    if not math.isclose(config["learning_rate"], learning_rate, rel_tol=1e-9):
        raise ValueError("Learning rate does not match declared arm")

    curve = []
    log_path = root / "steps.jsonl"
    if log_path.exists():
        with log_path.open() as source:
            for line in source:
                row = json.loads(line)
                step = row["optimizer_step"]
                if step != len(curve) + 1:
                    raise ValueError(f"Noncontiguous optimizer step {step}")
                rewards = row["rewards"]
                kl = row.get("sampled_kl")
                if len(rewards) != 2 or not all(math.isfinite(value) for value in rewards):
                    raise ValueError(f"Bad reward group at step {step}")
                if not isinstance(kl, (int, float)) or not math.isfinite(kl) or kl < -1e-5:
                    raise ValueError(f"Bad sampled KL at step {step}: {kl}")
                if len(row.get("scored_audio", [])) != 2:
                    raise ValueError(f"Missing scored audio binding at step {step}")
                for audio in row["scored_audio"]:
                    checked_audio(root, audio["path"], audio["sha256"])
                curve.append({"step": step, "reward": sum(rewards) / 2, "sampled_kl": kl})

    validation = {}
    for step in CHECKPOINTS:
        receipt_path = root / f"step_{step:06d}" / "receipt.json"
        if not receipt_path.is_file():
            continue
        receipt = json.loads(receipt_path.read_text())
        entries = receipt["heldout"]
        if receipt["optimizer_step"] != step or len(entries) != 59:
            raise ValueError(f"Incomplete 59-condition validation at step {step}")
        rewards = []
        peaks = []
        clips = []
        truncated = 0
        for index, entry in enumerate(entries):
            if entry["index"] != index or entry["seed"] != config["eval_seed_base"] + index:
                raise ValueError(f"Validation index/seed mismatch at step {step}, item {index}")
            checked_audio(root, entry["audio"], entry["audio_sha256"])
            checked_audio(root, entry["scored_audio"], entry["scored_audio_sha256"])
            if backend == "musecritic" and (entry["audio"] != entry["scored_audio"]
                                            or entry["audio_sha256"] != entry["scored_audio_sha256"]):
                raise ValueError(f"MuseCritic score/archive mismatch at step {step}, item {index}")
            rewards.append(entry["reward"]["mean"])
            peaks.append(entry["signal"]["peak"])
            clips.append(entry["signal"]["near_full_scale_fraction"])
            truncated += bool(entry["truncated"])
        mean = sum(rewards) / len(rewards)
        if not math.isclose(mean, receipt["mean_reward"], rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"Validation mean mismatch at step {step}")
        validation[str(step)] = {
            "mean_reward": mean,
            "n": len(entries),
            "max_peak": max(peaks),
            "mean_near_full_scale_fraction": sum(clips) / len(clips),
            "truncated": truncated,
        }
    if len(curve) == 100 and all(str(step) in validation for step in CHECKPOINTS):
        status = "verified-complete"
    else:
        status = "partial-verified"
    return {"status": status, "train_curve": curve, "validation": validation,
            "experiment_sha256": digest(config_path), "verified_train_steps": len(curve)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--reward-backend", choices=("songeval", "musecritic"), required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--compute-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.run_root, args.reward_backend, args.learning_rate, args.compute_dtype)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{result['status']}: {result.get('verified_train_steps', 0)} steps, "
          f"{len(result['validation'])} validation checkpoints")


if __name__ == "__main__":
    main()
