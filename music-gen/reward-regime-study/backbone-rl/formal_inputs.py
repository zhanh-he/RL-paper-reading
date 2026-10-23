"""Strict physical-line JSONL input validation for the frozen CMI protocol."""
import argparse
import hashlib
import json
from pathlib import Path

HASHES = {
    "train": "ff437d751d62742d92daf7ac56a678010c52d6c22def5624a8324301e6699be6",
    "valid": "ed5c5f36af48ef6ce81d15c7b15b429fb250ec4a52a04a43e55d96ec3cafb59d",
}


def read_jsonl(path):
    # splitlines() also splits valid Unicode lyric separators inside JSON strings.
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def check(train, valid):
    report = {}
    for split, path, expected_n in [("train", train, 270), ("valid", valid, 30)]:
        actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if actual != HASHES[split]:
            raise ValueError(f"Frozen {split} SHA mismatch: {actual}")
        rows = read_jsonl(path)
        if len(rows) != expected_n:
            raise ValueError(f"{split}: {len(rows)} records, expected {expected_n}")
        for row in rows:
            if not row.get("messages") or row["messages"][0].get("role") != "user":
                raise ValueError("Invalid native Muse messages")
            if [m['role'] for m in row['messages']] != ['user', 'assistant']:
                raise ValueError('This accepted colocate protocol requires one native user turn')
        report[split] = {"n": len(rows), "sha256": actual}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--reward", choices=["richness", "instrument", "arrangement", "mean7"])
    parser.add_argument("--steps", type=int, default=270)
    parser.add_argument("--lr", type=float, choices=[3e-4,1e-4], default=3e-4)
    args = parser.parse_args()
    data = check(args.train, args.valid)
    if args.manifest:
        if not args.reward or args.manifest.exists():
            raise ValueError("Fresh manifest and explicit reward required")
        record = {
            "model": "Muse-0.6B", "framework": "ms-swift/TRL + vLLM",
            "reward": args.reward, "lr": args.lr, "lr_scheduler": "constant",
            "beta": 0.05, "seed": 42, "steps": args.steps, "group_size": 2,
            "acceptance_only": args.steps != 270,
            "lora_rank": 8, "lora_alpha": 16, "target_modules": "all-linear backbone",
            "reference_kl": "sampled k3 over native completion-token masks",
            "data": data, "test_run": False,
            "scored_audio": "private PCM24-FLAC; public MP3 only",
            "technical_invalid_policy": "Mean7: under1s/silent finite audio => training reward0 with unavailable heads; no invented SongBench zero scores. Decoder invalid outcomes logged separately.",
        }
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(data), flush=True)


if __name__ == "__main__":
    main()
