"""Reference-backed SATB event rewards for training and adversarial audits."""

import argparse
import json
from pathlib import Path

from note_metrics import Note, VOICES, controlled_variants, score


WEIGHTS = {
    "onset": 0.25,
    "onset_offset": 0.25,
    "frame": 0.15,
    "coverage": 0.10,
    "weak_voice": 0.10,
    "precision": 0.10,
    "continuity": 0.05,
}


def components(reference, estimate):
    metrics = score(reference, estimate)
    return {
        "onset": metrics["macro"]["onset"],
        "onset_offset": metrics["macro"]["onset_offset"],
        "frame": metrics["macro"]["frame"],
        "coverage": metrics["active_voice"]["f1"],
        "weak_voice": min(metrics["per_voice"][v]["onset_offset"]["f1"] for v in VOICES),
        "precision": metrics["track_note"]["precision"],
        "continuity": 1 - metrics["fragmentation_rate"],
    }


def reward(reference, estimate, arm="combined"):
    values = components(reference, estimate)
    if arm == "combined":
        return sum(WEIGHTS[name] * value for name, value in values.items())
    if arm not in values:
        raise ValueError(f"Unknown reward arm: {arm}")
    return values[arm]


def load_json_notes(path):
    data = json.loads(Path(path).read_text())
    return [Note(int(n["pitch"]), float(n["start"]), float(n["end"]),
                 VOICES[n["voice"]] if isinstance(n["voice"], int) else n["voice"])
            for n in data["reference"]]


def audit(reference):
    rows = {}
    for name, notes in controlled_variants(reference).items():
        rows[name] = {
            "components": components(reference, notes),
            "combined": reward(reference, notes),
            "metrics": score(reference, notes),
        }
    return {
        "kind": "constructed_reward_counterexamples_not_grpo_outputs",
        "reference_note_count": len(reference),
        "reward_weights": WEIGHTS,
        "caveat": "These controlled edits expose reward loopholes; they do not prove an optimizer will discover them or that a weighted sum prevents all hacking.",
        "variants": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(load_json_notes(args.reference_json))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({name: {"combined": row["combined"], **row["components"]}
                      for name, row in result["variants"].items()}, indent=2))


if __name__ == "__main__":
    main()
