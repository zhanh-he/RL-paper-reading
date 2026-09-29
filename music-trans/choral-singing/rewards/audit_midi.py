"""Audit SATB rewards on paired local MIDI examples and controlled errors."""

import argparse
import json
import statistics
from pathlib import Path

from note_metrics import controlled_variants, load_satb_midi, score


METRICS = ("onset", "offset", "track_onset", "track_note", "active_voice")


def summarize(records):
    return {
        **{
            metric: {
                "mean_f1": statistics.mean(row[metric]["f1"] for row in records),
                "mean_precision": statistics.mean(row[metric]["precision"] for row in records),
                "mean_recall": statistics.mean(row[metric]["recall"] for row in records),
            }
            for metric in METRICS
        },
        "mean_fragmentation_rate": statistics.mean(row["fragmentation_rate"] for row in records),
        "total_reference_notes": sum(row["n_reference"] for row in records),
        "total_estimate_notes": sum(row["n_estimate"] for row in records),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    directory = Path(args.directory)
    real = []
    controlled = {}
    for gt_path in sorted(directory.glob("*_gt.mid")):
        estimate_path = directory / gt_path.name.replace("_gt.mid", "_pred.mid")
        if not estimate_path.is_file():
            continue
        reference = load_satb_midi(gt_path)
        estimate = load_satb_midi(estimate_path)
        real.append(score(reference, estimate))
        for name, notes in controlled_variants(reference).items():
            controlled.setdefault(name, []).append(score(reference, notes))
    if not real:
        raise ValueError("No paired *_gt.mid and *_pred.mid files found")
    result = {
        "status": "local_diagnostic_reward_audit_not_training",
        "pairs": len(real),
        "sampling": "existing examples; not a random held-out benchmark",
        "real_predictions": summarize(real),
        "controlled": {name: summarize(rows) for name, rows in controlled.items()},
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
