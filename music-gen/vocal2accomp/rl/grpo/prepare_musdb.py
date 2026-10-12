"""Deterministic, song-disjoint MUSDB18-HQ vocal/backing protocol.

All qualifying non-overlapping phrases are retained. Test PCM is never read by
this preparation command; only its song names and file headers are inventoried.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import soundfile as sf
import yaml

ERRATA = {
    "Al James - Schoolboy Facination": "synth vocals in other",
    "Chris Durban - Celebrate": "instrument bleed into vocals",
    "Detsky Sad - Walkie Talkie": "other sources in drums",
    "Enda Reilly - Cur An Long Ag Seol": "speech at start of other",
    "Hop Along - Sister Cities": "instrument bleed into vocals",
    "Leaf - Wicked": "guitar in drums at 2:20 and 3:02",
    "PR - Oh No": "left-channel mixture reconstruction mismatch",
}
PROMPT = "Instrumental accompaniment supporting the input singing, with coherent rhythm, harmony and expressive arrangement. No additional vocals."
STEMS = ("vocals", "drums", "bass", "other", "mixture")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def activity(x: np.ndarray, sr: int) -> float:
    hop = round(sr * 0.04)
    usable = x[:len(x) // hop * hop]
    energy = np.sqrt(np.mean(usable.reshape(-1, hop, x.shape[1]) ** 2, axis=(1, 2)))
    return float(np.mean(energy > 0.003))


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        for item in handle.infolist():
            target = (destination / item.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f"unsafe archive path {item.filename}")
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif not target.exists() or target.stat().st_size != item.file_size:
                handle.extract(item, destination)


def prepare(root: Path, config: Path, output: Path, seconds: float = 12.0) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    valid = set(yaml.safe_load(config.read_text())["validation_tracks"])
    songs = sorted((root / "train").iterdir())
    tests = sorted((root / "test").iterdir())
    if len(songs) != 100 or len(tests) != 50 or len(valid) != 14:
        raise ValueError("Expected complete MUSDB18-HQ and 14 official validation songs")
    if not valid.issubset({song.name for song in songs}):
        raise ValueError("Official validation songs missing")
    records, exclusions, inventory = [], [], []
    for song in songs + tests:
        split = "test" if song.parent.name == "test" else "valid" if song.name in valid else "train"
        headers = {stem: sf.info(song / f"{stem}.wav") for stem in STEMS}
        spec = {(h.frames, h.samplerate, h.channels) for h in headers.values()}
        if len(spec) != 1 or next(iter(spec))[1:] != (44100, 2):
            raise ValueError(f"unaligned or unexpected source headers: {song.name}")
        header = headers["vocals"]
        entry = {"song": song.name, "artist": song.name.split(" - ")[0], "split": split,
                 "duration": header.duration, "errata": ERRATA.get(song.name), "accepted": 0}
        inventory.append(entry)
        if split == "test":
            continue
        samples = round(seconds * header.samplerate)
        for start in range(0, header.frames - samples + 1, samples):
            audio = {stem: sf.read(song / f"{stem}.wav", start=start, frames=samples,
                                   dtype="float32", always_2d=True)[0] for stem in STEMS}
            if any(not np.isfinite(x).all() for x in audio.values()):
                raise ValueError(f"nonfinite source {song.name} @ {start}")
            vocal, backing = audio["vocals"], audio["drums"] + audio["bass"] + audio["other"]
            stem_activity = {stem: activity(audio[stem], 44100) for stem in STEMS[:4]}
            reasons = []
            if stem_activity["vocals"] < 0.35:
                reasons.append("vocal_activity_below_35pct")
            if activity(backing, 44100) < 0.5:
                reasons.append("backing_activity_below_50pct")
            if float(np.sqrt(np.mean(vocal.astype("float64") ** 2))) < 0.003:
                reasons.append("vocal_too_quiet")
            if float(np.mean(np.abs(vocal) >= 0.999)) > 0.001:
                reasons.append("clipped_vocal")
            # Whole affected recordings are placed in a separate challenge pool.
            if song.name in ERRATA:
                reasons.append("known_source_erratum")
            if reasons:
                exclusions.append({"song": song.name, "split": split, "start": start / 44100,
                                   "reasons": reasons})
                continue
            segment_id = hashlib.sha256(f"{song.name}:{start}:{samples}".encode()).hexdigest()[:16]
            dest = output / "clips" / split / segment_id
            dest.mkdir(parents=True, exist_ok=True)
            vocal_rms = float(np.sqrt(np.mean(vocal.astype("float64") ** 2)))
            joint_peak = max(float(np.max(np.abs(x))) for x in (vocal, backing, vocal + backing))
            gain = min(0.04 / max(vocal_rms, 1e-8), 0.90 / max(joint_peak, 1e-8), 4.0)
            vocal_path, backing_path = dest / "vocal.wav", dest / "backing.wav"
            sf.write(vocal_path, vocal * gain, 44100, subtype="PCM_24")
            sf.write(backing_path, backing * gain, 44100, subtype="FLOAT")
            record = {"id": segment_id, "song": song.name, "artist": entry["artist"],
                      "split": split, "start_seconds": start / 44100, "seconds": seconds,
                      "vocal": str(vocal_path.resolve()), "backing": str(backing_path.resolve()),
                      "text": PROMPT, "common_gain": gain, "stem_activity": stem_activity,
                      "vocal_sha256": file_hash(vocal_path), "backing_sha256": file_hash(backing_path)}
            records.append(record)
            entry["accepted"] += 1
        print(json.dumps({"event": "song", **entry}), flush=True)
    for split in ("train", "valid"):
        selected = [r for r in records if r["split"] == split]
        if not selected:
            raise ValueError(f"empty {split}")
        (output / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in selected))
    train = [r for r in records if r["split"] == "train"]
    valid_records = [r for r in records if r["split"] == "valid"]
    valid_songs = sorted({r["song"] for r in valid_records})
    # One fixed, earliest qualifying phrase per validation song for fast replay.
    probes = [next(r for r in valid_records if r["song"] == song) for song in valid_songs]
    (output / "valid-probes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in probes))
    train_artists = {r["artist"] for r in train}
    val_artists = {r["artist"] for r in valid_records}
    test_artists = {x["artist"] for x in inventory if x["split"] == "test"}
    summary = {"source": "https://zenodo.org/records/3338373", "protocol": "musdb-hq-v1",
               "seconds": seconds, "official_song_counts": {"train": 86, "valid": 14, "test": 50},
               "eligible": {split: {"songs": len({r["song"] for r in records if r["split"] == split}),
                                     "clips": sum(r["split"] == split for r in records),
                                     "minutes": sum(r["seconds"] for r in records if r["split"] == split) / 60}
                            for split in ("train", "valid")},
               "stem_activity_counts": {stem: sum(r["stem_activity"][stem] > 0.2 for r in train)
                                        for stem in ("drums", "bass", "other")},
               "artist_overlap": {"train_valid": sorted(train_artists & val_artists),
                                  "train_test": sorted(train_artists & test_artists)},
               "test_status": "sealed: no test PCM read; no test prompts or excerpts selected",
               "instrument_scope": "drums/bass/other activity only; other is not a fine instrument annotation",
               "pretraining_overlap": "unknown; no claim of pretraining-unseen songs",
               "train_manifest_sha256": file_hash(output / "train.jsonl"),
               "valid_manifest_sha256": file_hash(output / "valid.jsonl"),
               "config_sha256": file_hash(config), "inventory": inventory, "exclusions": exclusions}
    (output / "protocol.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--seconds", type=float, default=12)
    args = parser.parse_args()
    if args.archive:
        safe_extract(args.archive, args.root)
    result = prepare(args.root, args.config, args.output, args.seconds)
    print(json.dumps({k: v for k, v in result.items() if k not in ("inventory", "exclusions")}), flush=True)
    (args.output / "prepare.complete").touch()


if __name__ == "__main__":
    main()
