"""Cross-check SATB note F1 against mir_eval on a MIDI pair."""

import argparse
import json

import numpy as np
import mir_eval

from note_metrics import VOICES, load_satb_midi, score


def arrays(notes):
    intervals = np.asarray([(n.start, n.end) for n in notes], dtype=float).reshape(-1, 2)
    pitches = np.asarray([440 * 2 ** ((n.pitch - 69) / 12) for n in notes], dtype=float)
    return intervals, pitches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("estimate")
    args = parser.parse_args()
    reference = load_satb_midi(args.reference)
    estimate = load_satb_midi(args.estimate)
    ours = score(reference, estimate)
    results = {}
    for voice in VOICES:
        ref_intervals, ref_pitches = arrays([note for note in reference if note.voice == voice])
        est_intervals, est_pitches = arrays([note for note in estimate if note.voice == voice])
        results[voice] = {}
        for name, offset_ratio, ours_key in (
            ("onset", None, "onset"), ("onset_offset", 0.2, "onset_offset")
        ):
            _, _, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
                ref_intervals, ref_pitches, est_intervals, est_pitches,
                onset_tolerance=0.05, offset_ratio=offset_ratio,
                offset_min_tolerance=0.05,
            )
            ours_f1 = ours["per_voice"][voice][ours_key]["f1"]
            if abs(f1 - ours_f1) > 1e-8:
                raise AssertionError(f"{voice} {name}: mir_eval {f1}, ours {ours_f1}")
            results[voice][name] = f1
    print(json.dumps({"status": "matched_mir_eval", "per_voice_f1": results}, indent=2))


if __name__ == "__main__":
    main()
