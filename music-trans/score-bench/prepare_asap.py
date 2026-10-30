"""Download a work-disjoint, deterministic ASAP subset using ACPAS split metadata."""
import argparse
import csv
import hashlib
import io
import json
import urllib.parse
import urllib.request
from pathlib import Path

REVISION = "8cba199e15931975542010a7ea2ff94a6fc9cbee"
BASE = f"https://raw.githubusercontent.com/TimFelixBeyer/asap-dataset/{REVISION}/"
METADATA = "https://cheriell.github.io/research/ACPAS_dataset/metadata_R.csv"


def download(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        with urllib.request.urlopen(url, timeout=60) as response:
            destination.write_bytes(response.read())
    return hashlib.sha256(destination.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output
    download(METADATA, root / "metadata_R.csv")
    download(BASE + "metadata.csv", root / "asap-metadata.csv")
    asap = {r["midi_performance"]: r for r in csv.DictReader(io.StringIO((root / "asap-metadata.csv").read_text()))}
    source = list(csv.DictReader(io.StringIO((root / "metadata_R.csv").read_text())))
    records = []
    used = set()
    for split, count in [("train", 16), ("validation", 4), ("test", 8)]:
        candidates = [r for r in source if r["source"] == "ASAP" and r["split"] == split and r["aligned"].lower() == "true"]
        candidates.sort(key=lambda r: hashlib.sha256(r["performance_MIDI_external"].encode()).hexdigest())
        selected = 0
        for row in candidates:
            relative = row["performance_MIDI_external"].replace("{ASAP}/", "")
            entry = asap.get(relative)
            if not entry or entry["folder"] in used:
                continue
            files = {}
            for field in ["midi_performance", "xml_score", "midi_score", "performance_annotations", "midi_score_annotations"]:
                value = entry[field]
                files[field] = {"path": value, "sha256": download(BASE + urllib.parse.quote(value), root / value)}
            used.add(entry["folder"])
            record = {"id": f"{split}-{selected:02d}", "split": split, "composer": row["composer"], "work": entry["folder"], "files": files}
            records.append(record)
            print(json.dumps({"id": record["id"], "work": record["work"]}), flush=True)
            selected += 1
            if selected == count:
                break
        if selected != count:
            raise RuntimeError(f"Only {selected}/{count} works available for {split}")
    manifest = {"dataset": "ASAP", "asap_revision": REVISION, "split_source": METADATA,
                "protocol": "One performance per work; source ACPAS real split; SHA256-sorted subset, selected before inference. DuET checkpoint pretraining overlap is not independently audited.", "samples": records}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
