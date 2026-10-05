"""Freeze a decontaminated lyrics-to-song prompt split from pinned source data.

Raw lyrics and the generated training manifest stay under ignored datasets/.
The public summary contains counts and hashes, but no source lyrics.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


SOURCES = {
    "cmi_train": ("cmi-pref/cmi_train.jsonl", "fd157070f09ad51d807f4b447bf5a9668fd9b00890c7d646ffd1331850e7d8d0"),
    "cmi_test": ("cmi-pref/cmi_test.jsonl", "824a92f33d04096e739fdc771d09bda125943baafae5c0eb5b6e9e267d3762e9"),
    "wildsongbench": ("wildsongbench/reproduction_manifest.jsonl", "859e5225d319912efb64d587d1d5eb6cf0abc9fb162cc3d5f02d1688cff2edc0"),
}
REVISIONS = {
    "cmi_pref": "5282fbe784e326014894b299cb22645cd7d56057",
    "wildsongbench": "e361b85a8d7365d3079fa9647641a87ce3c8c5b5",
}


def normalized(text):
    return " ".join(text.split()).casefold()


def rows(path):
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(root):
    loaded = {}
    for name, (relative, expected) in SOURCES.items():
        path = root / relative
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"{path}: SHA256 {actual} != pinned {expected}")
        loaded[name] = rows(path)

    cmi_train = loaded["cmi_train"]
    cmi_test = loaded["cmi_test"]
    wsb = loaded["wildsongbench"]
    if (len(cmi_train), len(cmi_test), len(wsb)) != (3527, 500, 192):
        raise ValueError("Unexpected official split size")
    test_lyrics = {normalized(row["lyrics"]) for row in cmi_test if row["lyrics"].strip()}
    wsb_lyrics = {normalized(row["lyrics"]) for row in wsb if row["lyrics"].strip()}

    counts = Counter()
    unique = {}
    for row in cmi_train:
        if not row["lyrics"].strip():
            counts["no_lyrics"] += 1
            continue
        if row["ref-audio-path"].strip():
            counts["requires_reference_audio"] += 1
            continue
        counts["compatible_votes"] += 1
        lyric_key = normalized(row["lyrics"])
        if lyric_key in test_lyrics:
            counts["test_lyrics_overlap_votes"] += 1
            continue
        if lyric_key in wsb_lyrics:
            counts["wildsongbench_lyrics_overlap_votes"] += 1
            continue
        condition = (normalized(row["prompt"]), lyric_key)
        condition_id = hashlib.sha256(json.dumps(condition, ensure_ascii=False).encode()).hexdigest()
        if condition_id in unique:
            counts["duplicate_condition_votes"] += 1
            unique[condition_id]["source_prompt_ids"].add(str(row["prompt id"]))
            continue
        unique[condition_id] = {
            "condition_id": condition_id,
            "style": row["prompt"].strip(),
            "lyrics": row["lyrics"].strip(),
            "source_prompt_ids": {str(row["prompt id"])},
        }
    if len(unique) != 293:
        raise ValueError(f"Unexpected decontaminated condition count: {len(unique)}")

    # Hash-order split is independent of source row order and repeated votes.
    ordered = sorted(unique.values(), key=lambda row: row["condition_id"])
    validation_count = round(len(ordered) * 0.2)
    valid_ids = {row["condition_id"] for row in ordered[:validation_count]}
    train = []
    valid = []
    for row in sorted(unique.values(), key=lambda item: item["condition_id"]):
        row["source_prompt_ids"] = sorted(row["source_prompt_ids"])
        (valid if row["condition_id"] in valid_ids else train).append(row)

    manifest = {
        "protocol": "cmi-pref-lyrics-no-ref-decontaminated-v1",
        "source_revisions": REVISIONS,
        "source_sha256": {name: expected for name, (_, expected) in SOURCES.items()},
        "split_rule": "exact normalized lyrics exclusion against all CMI-Pref test and WildSongBench; dedupe normalized style+lyrics; first 20% by SHA256 condition_id validation",
        "train": train,
        "valid": valid,
        "sealed_cmi_test_prompt_ids": sorted({str(row["prompt id"]) for row in cmi_test}),
        "sealed_wildsongbench_prompt_indices": sorted(int(row["prompt_index"]) for row in wsb),
    }
    output = root / "formal-lyrics2song-v1.json"
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "protocol": manifest["protocol"],
        "source_revisions": REVISIONS,
        "source_sha256": manifest["source_sha256"],
        "manifest_sha256": sha256(output),
        "counts": {
            "cmi_pref_train_votes": len(cmi_train),
            "cmi_pref_test_votes_sealed": len(cmi_test),
            "wildsongbench_test_prompts_sealed": len(wsb),
            **dict(counts),
            "wildsongbench_lyrics_overlap_votes": counts["wildsongbench_lyrics_overlap_votes"],
            "unique_eligible_conditions": len(unique),
            "train_conditions": len(train),
            "validation_conditions": len(valid),
        },
        "limits": [
            "CMI-Pref eligible lyrics-only subset, not full 3527-row train set.",
            "CMI-Pref test and WildSongBench remain sealed; exact normalized lyrics checks do not exclude semantic paraphrases.",
            "Existing 3-prompt demo is illustrative only and is not the formal validation curve.",
        ],
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    summary = prepare(args.data_root)
    encoded = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(encoded, encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
