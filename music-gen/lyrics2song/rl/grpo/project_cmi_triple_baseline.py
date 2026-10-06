"""Project the frozen three-part CMI split to the models' text+lyrics interface."""

import argparse
import hashlib
import json
from pathlib import Path


SOURCE_SHA256 = "b611333ef0abdeaf7a0473ea0beaa470673c3414f05b816a97904d2ade25b2e0"
SOURCE_PROTOCOL = "cmi-pref-text-audio-lyrics-decontaminated-v1"
PROTOCOL = "cmi-pref-triple-source-text-lyrics-baseline-v1"


def project(source_path, output_path):
    source_bytes = source_path.read_bytes()
    actual_sha = hashlib.sha256(source_bytes).hexdigest()
    if actual_sha != SOURCE_SHA256:
        raise ValueError(f"Frozen three-part manifest changed: {actual_sha}")
    source = json.loads(source_bytes)
    if source.get("protocol") != SOURCE_PROTOCOL:
        raise ValueError("Unexpected source protocol")

    def conditions(split):
        return [{"condition_id": row["condition_id"], "style": row["style"],
                 "lyrics": row["lyrics"]} for row in source[split]]

    manifest = {
        "protocol": PROTOCOL,
        "source_protocol": SOURCE_PROTOCOL,
        "source_manifest_sha256": SOURCE_SHA256,
        "input_modalities": ["text", "lyrics"],
        "reference_audio_used": False,
        "note": "Train/valid membership comes from three-part conditions; source reference audio is reserved for future models, not passed to this baseline.",
        "train": conditions("train"),
        "valid": conditions("valid"),
    }
    if (len(manifest["train"]), len(manifest["valid"])) != (240, 60):
        raise ValueError("Unexpected projected split size")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return hashlib.sha256(output_path.read_bytes()).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("datasets/cmi-pref-triple-v1.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("datasets/cmi-pref-triple-text-lyrics-baseline-v1.json"))
    arguments = parser.parse_args()
    print(project(arguments.source, arguments.output))
