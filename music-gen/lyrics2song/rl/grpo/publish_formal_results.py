"""Merge verified, lyrics-free run receipts into the public formal demo manifest."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary", action="append", required=True,
                        help="ARM=PATH to a collect_formal_run.py output")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    for item in args.summary:
        arm_name, separator, path = item.partition("=")
        if not separator or arm_name not in manifest["arms"]:
            parser.error(f"Unknown arm or malformed --summary: {item}")
        summary = json.loads(Path(path).read_text())
        if summary["status"] not in ("partial-verified", "verified-complete"):
            parser.error(f"Summary is not verified: {path}")
        if summary["verified_train_steps"] != len(summary["train_curve"]):
            parser.error(f"Inconsistent step count: {path}")
        if any(checkpoint not in manifest["checkpoint_steps"]
               for checkpoint in map(int, summary["validation"])):
            parser.error(f"Unexpected validation checkpoint: {path}")
        manifest["arms"][arm_name].update(summary)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
