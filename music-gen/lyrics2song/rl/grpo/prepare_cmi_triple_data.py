"""Freeze a metadata-only CMI-Pref text+reference-audio+lyrics split.

The private condition manifest stays in ignored datasets/. No reference audio
is downloaded, and this script does not make YuE2 or Muse audio-conditioned.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from prepare_formal_data import REVISIONS, SOURCES, normalized, rows, sha256


PROTOCOL = "cmi-pref-text-audio-lyrics-decontaminated-v1"


def condition_key(row):
    return (normalized(row["prompt"]), normalized(row["lyrics"]), row["ref-audio-path"].strip())


def condition_id(key):
    payload = json.dumps(key, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prepare(root, summary_path=None):
    loaded = {}
    for name, (relative, expected) in SOURCES.items():
        path = root / relative
        if sha256(path) != expected:
            raise ValueError(f"Pinned SHA256 mismatch: {path}")
        loaded[name] = rows(path)
    train_rows, test_rows = loaded["cmi_train"], loaded["cmi_test"]
    if (len(train_rows), len(test_rows), len(loaded["wildsongbench"])) != (3527, 500, 192):
        raise ValueError("Unexpected source split size")

    eligible = lambda row: bool(row["prompt"].strip() and row["lyrics"].strip()
                                and row["ref-audio-path"].strip())
    train_votes = [row for row in train_rows if eligible(row)]
    test_votes = [row for row in test_rows if eligible(row)]
    if (len(train_votes), len(test_votes)) != (469, 125):
        raise ValueError("Unexpected triple-modality vote count")
    test_lyrics = {normalized(row["lyrics"]) for row in test_rows if row["lyrics"].strip()}
    wsb_lyrics = {normalized(row["lyrics"]) for row in loaded["wildsongbench"]
                  if row["lyrics"].strip()}
    clean_votes = [row for row in train_votes
                   if normalized(row["lyrics"]) not in test_lyrics | wsb_lyrics]
    if len(clean_votes) != 435:
        raise ValueError("Unexpected decontaminated vote count")

    unique = {}
    for row in clean_votes:
        key = condition_key(row)
        cid = condition_id(key)
        if cid not in unique:
            unique[cid] = {
                "condition_id": cid,
                "style": row["prompt"].strip(),
                "lyrics": row["lyrics"].strip(),
                "ref_audio_path": row["ref-audio-path"].strip(),
                "source_prompt_ids": [],
                "source_vote_count": 0,
            }
        unique[cid]["source_vote_count"] += 1
        unique[cid]["source_prompt_ids"].append(str(row["prompt id"]))
    if len(unique) != 300:
        raise ValueError("Unexpected independent condition count")

    groups = defaultdict(list)
    for condition in unique.values():
        condition["source_prompt_ids"] = sorted(set(condition["source_prompt_ids"]))
        groups[normalized(condition["lyrics"])].append(condition)
    ordered_groups = sorted(groups, key=lambda lyric: hashlib.sha256(lyric.encode()).hexdigest())
    valid_groups = set()
    valid_count = 0
    for lyric in ordered_groups:
        if valid_count + len(groups[lyric]) <= 60:
            valid_groups.add(lyric)
            valid_count += len(groups[lyric])
    if valid_count != 60:
        raise ValueError("Cannot make an exact 60-condition lyric-group validation split")
    train = sorted((item for lyric, items in groups.items() if lyric not in valid_groups
                    for item in items), key=lambda item: item["condition_id"])
    valid = sorted((item for lyric, items in groups.items() if lyric in valid_groups
                    for item in items), key=lambda item: item["condition_id"])
    if (len(train), len(valid)) != (240, 60):
        raise ValueError("Unexpected train/validation condition counts")

    manifest = {
        "protocol": PROTOCOL,
        "status": "data-frozen-model-audio-interface-unverified-do-not-train",
        "source_revisions": REVISIONS,
        "source_sha256": {name: expected for name, (_, expected) in SOURCES.items()},
        "split_rule": "exclude normalized lyrics in all official CMI-Pref test and WildSongBench; dedupe normalized text+lyrics+reference path; SHA256-order lyric groups into 60 valid conditions; 240 train",
        "train": train,
        "valid": valid,
    }
    output = root / "cmi-pref-triple-v1.json"
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    test_keys = {condition_key(row) for row in test_votes}
    train_keys = {condition_key(row) for row in train_votes}
    summary = {
        "protocol": PROTOCOL,
        "status": manifest["status"],
        "source_revisions": REVISIONS,
        "source_sha256": manifest["source_sha256"],
        "manifest_sha256": sha256(output),
        "counts": {
            "official_train_votes": 469,
            "official_test_votes": 125,
            "official_train_conditions": len(train_keys),
            "official_test_conditions": len(test_keys),
            "exact_official_train_test_condition_overlap": len(train_keys & test_keys),
            "excluded_train_votes_with_sealed_lyrics": len(train_votes) - len(clean_votes),
            "decontaminated_train_votes": len(clean_votes),
            "decontaminated_train_conditions": len(unique),
            "frozen_train_conditions": len(train),
            "frozen_validation_conditions": len(valid),
            "wildsongbench_additional_lyric_overlap_votes": sum(
                normalized(row["lyrics"]) in wsb_lyrics
                and normalized(row["lyrics"]) not in test_lyrics for row in train_votes),
        },
        "limits": [
            "Metadata and official reference-audio file listing checked; MP3 payloads not downloaded or decoded.",
            "No native external-reference-audio conditioning verified in the current YuE2 or Muse GRPO code.",
            "Official preference test supplies conditions, not a unique target recording for new generations.",
        ],
    }
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", type=Path, default=Path("datasets"))
    parser.add_argument("--summary-output", type=Path,
                        default=Path("platform/site/demos/cmi-triple-data-summary.json"))
    arguments = parser.parse_args()
    print(json.dumps(prepare(arguments.datasets_root, arguments.summary_output), indent=2))
