"""CPU metric controls on artificial notes, not a model-quality experiment."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import mir_eval
import numpy as np


def hz(midi):
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


def measure(ref_intervals, ref_midi, intervals, midi, *, folded, offsets):
    reference = hz(ref_midi % 12 + 60 if folded else ref_midi)
    prediction = hz(midi % 12 + 60 if folded else midi)
    _, _, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, reference, intervals, prediction,
        onset_tolerance=0.05, pitch_tolerance=50.0,
        offset_ratio=0.2 if offsets else None, offset_min_tolerance=0.05,
    )
    return round(float(f1) * 100, 8)


def run():
    midi = np.array([60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79], dtype=float)
    onsets = 1.0 + np.arange(len(midi)) * 0.8
    reference = np.column_stack((onsets, onsets + 0.3))
    cases = {
        "unchanged": (reference.copy(), midi.copy()),
        "all_notes_up_one_octave": (reference.copy(), midi + 12),
        "double_all_durations": (np.column_stack((onsets, onsets + 0.6)), midi.copy()),
        "alternating_octave_errors": (reference.copy(), midi + (np.arange(len(midi)) % 2) * 12),
        "all_events_delayed_100ms": (reference + 0.1, midi.copy()),
    }
    rows = []
    for name, (intervals, pitches) in cases.items():
        scores = {
            "pitch_class_onset_f1_percent": measure(reference, midi, intervals, pitches, folded=True, offsets=False),
            "absolute_pitch_onset_f1_percent": measure(reference, midi, intervals, pitches, folded=False, offsets=False),
            "absolute_pitch_onset_offset_f1_percent": measure(reference, midi, intervals, pitches, folded=False, offsets=True),
        }
        rows.append({"case": name, **scores, "estimated_intervals_seconds": intervals.tolist(), "estimated_midi_pitches": pitches.tolist()})
    expected = {
        "unchanged": (100, 100, 100),
        "all_notes_up_one_octave": (100, 0, 0),
        "double_all_durations": (100, 100, 0),
        "alternating_octave_errors": (100, 50, 50),
        "all_events_delayed_100ms": (0, 0, 0),
    }
    score_fields = ["pitch_class_onset_f1_percent", "absolute_pitch_onset_f1_percent", "absolute_pitch_onset_offset_f1_percent"]
    for row in rows:
        assert tuple(row[field] for field in score_fields) == expected[row["case"]], row
    return {
        "experiment": "E0_transcription_metric_controls",
        "scope": "Artificial note arrays only; no audio or transcription model was run",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ["mir_eval", "numpy", "scipy"]},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "metric": {"onset_tolerance_seconds": 0.05, "pitch_tolerance_cents": 50, "offset_ratio": 0.2, "minimum_offset_tolerance_seconds": 0.05, "matching_library": "mir_eval.transcription.precision_recall_f1_overlap", "aggregate": "one synthetic sequence; not a dataset average"},
        "reference_intervals_seconds": reference.tolist(),
        "reference_midi_pitches": midi.tolist(),
        "checks_passed": len(expected),
        "rows": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"scope": result["scope"], "checks_passed": result["checks_passed"], "rows": [{k:v for k,v in row.items() if not k.startswith("estimated_")} for row in result["rows"]]}, indent=2))
