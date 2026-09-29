"""Score an official MuScriptor-demo MIDI on the original synthetic SATB example."""

import argparse
import json
from pathlib import Path

import pretty_midi

from evaluate_notes import counts, prf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notes", type=Path, required=True)
    parser.add_argument("--midi", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(args.notes.read_text())["reference"]
    midi = pretty_midi.PrettyMIDI(str(args.midi))
    estimated = [
        {"pitch": int(note.pitch), "start": float(note.start), "end": float(note.end)}
        for instrument in midi.instruments for note in instrument.notes
    ]
    receipt = {
        "status": "official_online_demo_baseline_not_grpo",
        "input": "Original 10.2-second harmonic SATB synthesis; not a held-out benchmark",
        "model": "MuScriptor official hosted demo",
        "instrument_tracks": [
            {"name": instrument.name, "program": int(instrument.program),
             "notes": len(instrument.notes)} for instrument in midi.instruments
        ],
        "reference_notes": len(reference),
        "predicted_notes": len(estimated),
        "estimated_notes": estimated,
        "protocol": "mir_eval onset tolerance 50ms; offset tolerance max(50ms, 20% reference duration)",
        "note_onset_50ms": prf(counts(reference, estimated, track_aware=False, require_offset=False)),
        "note_onset_offset_50ms": prf(counts(reference, estimated, track_aware=False, require_offset=True)),
        "track_aware_metrics": None,
        "caveat": "The hosted model returned one acoustic-piano track without SATB voice labels; do not compare a track-aware F1. This output cannot establish a MuScriptor GRPO result.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
