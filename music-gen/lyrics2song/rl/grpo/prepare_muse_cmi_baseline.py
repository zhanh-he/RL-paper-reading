"""Create Muse chat inputs from the frozen CMI three-part text+lyrics projection."""

import argparse
import hashlib
import json
from pathlib import Path


PROJECTED_SHA256 = "b061588397d54177b25b678962caf756771498b927b9931542908c5e64a7e109"


def prompt(row):
    style = row["style"].strip()
    lyrics = row["lyrics"].strip()
    return (f"Please generate a song in the following style: {style}.\n"
            "Next, I will tell you the requirements and lyrics for the song fragment to be generated.\n"
            f"[Verse][desc:{style}][lyrics:\n{lyrics}]")


def prepare(source_path, output_dir):
    actual = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if actual != PROJECTED_SHA256:
        raise ValueError(f"Projected manifest SHA256 changed: {actual}")
    manifest = json.loads(source_path.read_text())
    if manifest.get("input_modalities") != ["text", "lyrics"] or manifest.get("reference_audio_used") is not False:
        raise ValueError("Muse inputs must remain an explicitly text+lyrics-only baseline")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {"source_manifest_sha256": actual, "input_modalities": ["text", "lyrics"],
              "reference_audio_used": False, "files": {}}
    for split, expected in (("train", 240), ("valid", 60)):
        conditions = manifest[split]
        if len(conditions) != expected:
            raise ValueError(f"Unexpected {split} condition count")
        path = output_dir / f"muse-cmi-triple-{split}.jsonl"
        with path.open("w", encoding="utf-8") as output:
            for condition in conditions:
                messages = [{"role": "user", "content": prompt(condition)},
                            {"role": "assistant", "content": ""}]
                output.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
        result["files"][split] = {"rows": expected,
                                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("datasets/cmi-pref-triple-text-lyrics-baseline-v1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/muse-cmi-triple-baseline"))
    arguments = parser.parse_args()
    print(json.dumps(prepare(arguments.source, arguments.output_dir), indent=2))
