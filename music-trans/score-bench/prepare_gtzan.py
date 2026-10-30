"""Package one real GTZAN mini recording per genre with paper-matched annotations."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    samples = []
    for genre in sorted((args.audio / "genres").iterdir()):
        if not genre.is_dir():
            continue
        for audio in sorted(genre.glob("*.wav")):
            identity = "gtzan_" + audio.stem.replace(".", "_")
            annotation = args.annotations / "gtzan/annotations/beats" / (identity + ".beats")
            if not annotation.is_file():
                continue
            shutil.copy2(audio, args.output / audio.name)
            shutil.copy2(annotation, args.output / annotation.name)
            samples.append({"id": identity, "genre": genre.name, "audio": audio.name,
                "annotations": annotation.name, "audio_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
                "annotation_sha256": hashlib.sha256(annotation.read_bytes()).hexdigest()})
            break
    if len(samples) != 10:
        raise RuntimeError(f"Expected ten genres, got {len(samples)}")
    (args.output / "manifest.json").write_text(json.dumps({"selection": "First lexicographic available annotated recording per genre, fixed before inference", "samples": samples}, indent=2) + "\n")


if __name__ == "__main__":
    main()
