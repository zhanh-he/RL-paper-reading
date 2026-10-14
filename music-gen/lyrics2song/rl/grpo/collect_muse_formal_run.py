"""Recover ms-swift Muse curves, checkpoints, and audio bindings, including failures."""

import argparse
import hashlib
import json
import math
from pathlib import Path


TRAIN_SHA = "ff437d751d62742d92daf7ac56a678010c52d6c22def5624a8324301e6699be6"
SOURCE_SHA = "2f5f31219514ad479984e921a8dc6603680e275f31531778c7bb018da5df0136"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def collect(root):
    experiment = json.loads((root / "experiment.json").read_text())
    expected = {"muse_train_jsonl_sha256": TRAIN_SHA, "source_split_sha256": SOURCE_SHA,
                "train_conditions": 270, "valid_conditions": 30,
                "input_modalities": ["text", "lyrics"], "reference_audio_used": False}
    for key, value in expected.items():
        if experiment.get(key) != value:
            raise ValueError(f"Muse protocol mismatch: {key}")
    logs = list(root.glob("checkpoint/v*/logging.jsonl"))
    if len(logs) != 1:
        raise ValueError(f"Expected one training log, found {len(logs)}")
    resume_step = experiment.get('resume_step', 0)
    prefix = collect(Path(experiment['resume_checkpoint']).parents[2]) if resume_step else None
    curve = prefix['train_curve'][:resume_step] if prefix else []
    for line in logs[0].read_text().splitlines():
        row = json.loads(line)
        if "reward" not in row:
            continue
        step = int(row["global_step/max_steps"].split("/")[0])
        if step != len(curve) + 1:
            raise ValueError(f"Noncontiguous Muse step {step}")
        if not all(math.isfinite(row[k]) for k in ("reward", "kl")):
            raise ValueError(f"Nonfinite Muse curve at {step}")
        curve.append({"step": step, "reward": row["reward"], "sampled_kl": row["kl"]})
    rollouts = [json.loads(line) for line in (root / "rollouts.jsonl").read_text().splitlines()]
    if len(rollouts) != len(curve) - resume_step:
        raise ValueError("Training curve and rollout count differ")
    recorded = {}
    invalid_path = root / 'invalid_outputs.jsonl'
    invalid = {r['request_id']: r for r in (json.loads(line) for line in invalid_path.read_text().splitlines())} if invalid_path.exists() else {}
    if experiment["reward"] == "songeval":
        for path in root.glob("songeval_scoring/*/bindings.jsonl"):
            for line in path.read_text().splitlines():
                binding = json.loads(line)
                recorded[binding["audio_path"]] = binding["audio_sha256"]
    else:
        receipt = root / "receipt.json"
        if receipt.exists():
            for binding in json.loads(receipt.read_text())["bound_audio"]:
                recorded[str(root / binding["audio_path"])] = binding["sha256"]
    bindings = [r for r in prefix['audio_bindings'] if r['step'] <= resume_step] if prefix else []
    for step, (row, point) in enumerate(zip(rollouts, curve[resume_step:]), resume_step + 1):
        rewards = []
        for index in (0, 1):
            audio_text = row[f"audio_path_{index}"]
            audio = Path(audio_text)
            reward = row[f"reward_{index}"]
            if not audio_text:
                request_id = row[f'completion_{index}_request_id']
                rejection = invalid.get(request_id)
                if reward != 0 or not rejection or rejection['reward'] != 0:
                    raise ValueError(f'Unexplained missing audio at step {step}, rollout {index}')
                bindings.append({'step': step, 'rollout': index, 'status': 'invalid-output-zero',
                                 'reason': rejection['reason'], 'invalid_receipt_sha256': digest(invalid_path)})
                rewards.append(reward)
                continue
            if not audio.is_file() or not math.isfinite(reward):
                raise ValueError(f"Missing audio or reward at step {step}, rollout {index}")
            actual = digest(audio)
            if str(audio) in recorded and actual != recorded[str(audio)]:
                raise ValueError(f"Audio hash mismatch at step {step}")
            bindings.append({"step": step, "rollout": index, "sha256": actual,
                             "recorded_hash_checked": str(audio) in recorded})
            rewards.append(reward)
        if not math.isclose(sum(rewards) / 2, point["reward"], abs_tol=1e-5):
            raise ValueError(f"Rollout and logged reward differ at step {step}")
    checkpoints = sorted({int(path.parent.name.removeprefix("checkpoint-"))
                          for path in root.rglob("adapter_model.safetensors")})
    if prefix:
        checkpoints = sorted(set(checkpoints) | {s for s in prefix['checkpoints'] if s <= resume_step})
    complete = len(curve) == 270 and 270 in checkpoints
    errors = [line.strip() for line in (root / "train.log").read_text(errors="replace").split("\n")
              if any(key in line for key in ("ValueError:", "RuntimeError:", "OutOfMemoryError:"))]
    if not complete and not errors:
        raise ValueError("Incomplete training has no captured failure reason")
    if errors and "no <AUDIO_*>" in errors[-1]:
        failure = "Generated completion contained no AUDIO tokens; strict reward decoding aborted."
    elif errors and "Padding size should be less" in errors[-1]:
        failure = "SongEval/MuQ could not process a very short generated clip (960 samples)."
    else:
        failure = errors[-1][-350:] if errors else None
    return {
        "status": "verified-training-complete" if complete else "verified-partial-failed",
        "model": "Muse", "reward_backend": experiment["reward"],
        "learning_rate": experiment["learning_rate"], "train_conditions": 270,
        "valid_conditions": 30, "verified_train_steps": len(curve), "train_curve": curve,
        "checkpoints": checkpoints, "failure_reason": failure, "validation": {},
        "audio_bindings": bindings, "experiment_sha256": digest(root / "experiment.json"),
        "invalid_output_count": sum(r.get('status') == 'invalid-output-zero' for r in bindings),
        "validation_status": "not-yet-evaluated",
        "resume_step": resume_step,
        "restores_optimizer_scheduler_rng": experiment.get('restores_optimizer_scheduler_rng', False),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.run_root)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"], result["verified_train_steps"], result["checkpoints"], flush=True)


if __name__ == "__main__":
    main()
