"""Project the frozen three-part CMI split to the models' text+lyrics interface."""

import argparse
import hashlib
import json
from pathlib import Path


SOURCE_SHA256 = "b611333ef0abdeaf7a0473ea0beaa470673c3414f05b816a97904d2ade25b2e0"
SOURCE_SHA256_9TO1 = "2f5f31219514ad479984e921a8dc6603680e275f31531778c7bb018da5df0136"
SOURCE_PROTOCOL = "cmi-pref-text-audio-lyrics-decontaminated-v1"
PROTOCOL = "cmi-pref-triple-source-text-lyrics-baseline-v1"
PROTOCOL_9TO1 = "cmi-pref-triple-source-text-lyrics-baseline-9to1-v2"


def project(source_path, output_path, validation_conditions=60):
    if validation_conditions not in (30, 60):
        raise ValueError("Unsupported validation split")
    expected_source_sha = SOURCE_SHA256_9TO1 if validation_conditions == 30 else SOURCE_SHA256
    source_bytes = source_path.read_bytes()
    actual_sha = hashlib.sha256(source_bytes).hexdigest()
    if actual_sha != expected_source_sha:
        raise ValueError(f"Frozen three-part manifest changed: {actual_sha}")
    source = json.loads(source_bytes)
    if source.get("protocol") != SOURCE_PROTOCOL:
        raise ValueError("Unexpected source protocol")

    def conditions(split):
        return [{"condition_id": row["condition_id"], "style": row["style"],
                 "lyrics": row["lyrics"]} for row in source[split]]

    manifest = {
        "protocol": PROTOCOL_9TO1 if validation_conditions == 30 else PROTOCOL,
        "source_protocol": SOURCE_PROTOCOL,
        "source_manifest_sha256": expected_source_sha,
        "input_modalities": ["text", "lyrics"],
        "reference_audio_used": False,
        "note": "Train/valid membership comes from three-part conditions; source reference audio is reserved for future models, not passed to this baseline.",
        "train": conditions("train"),
        "valid": conditions("valid"),
    }
    if (len(manifest["train"]), len(manifest["valid"])) != (300 - validation_conditions, validation_conditions):
        raise ValueError("Unexpected projected split size")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return hashlib.sha256(output_path.read_bytes()).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("datasets/cmi-pref-triple-v1.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("datasets/cmi-pref-triple-text-lyrics-baseline-v1.json"))
    parser.add_argument("--validation-conditions", type=int, choices=(30, 60), default=60)
    arguments = parser.parse_args()
    print(project(arguments.source, arguments.output, arguments.validation_conditions))
