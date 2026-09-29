"""Evaluate ChoralStream MIDI decoding with mir_eval-style 50 ms tolerances.

The frame-head GRPO pilot does not update the AR event decoder. This script
checks that parameter boundary, verifies one paired decode, and evaluates the
frozen event decoder on the same held-out song IDs used by the frame pilot.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import mir_eval.transcription
import numpy as np
import torch


def counts(reference, estimate, *, track_aware, require_offset):
    def match_voice(ref, est):
        ref_intervals = np.asarray([[row["start"], row["end"]] for row in ref], dtype=float).reshape(-1, 2)
        est_intervals = np.asarray([[row["start"], row["end"]] for row in est], dtype=float).reshape(-1, 2)
        ref_pitches = 440.0 * 2.0 ** ((np.asarray([row["pitch"] for row in ref], dtype=float) - 69) / 12)
        est_pitches = 440.0 * 2.0 ** ((np.asarray([row["pitch"] for row in est], dtype=float) - 69) / 12)
        matches = mir_eval.transcription.match_notes(
            ref_intervals, ref_pitches, est_intervals, est_pitches,
            onset_tolerance=0.05,
            offset_ratio=0.2 if require_offset else None,
            offset_min_tolerance=0.05,
        )
        return len(matches), len(est) - len(matches), len(ref) - len(matches)

    if not track_aware:
        return match_voice(reference, estimate)
    parts = [
        match_voice([row for row in reference if row["voice"] == voice],
                    [row for row in estimate if row["voice"] == voice])
        for voice in range(4)
    ]
    return tuple(sum(part[index] for part in parts) for index in range(3))


def prf(count):
    tp, fp, fn = count
    return {
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "tp": tp, "fp": fp, "fn": fn,
    }


def add(left, right):
    return tuple(a + b for a, b in zip(left, right))


def score_events(reference, estimate):
    return {
        "note_onset_50ms": counts(reference, estimate, track_aware=False, require_offset=False),
        "note_onset_offset_50ms": counts(reference, estimate, track_aware=False, require_offset=True),
        "track_note_onset_50ms": counts(reference, estimate, track_aware=True, require_offset=False),
        "track_note_onset_offset_50ms": counts(reference, estimate, track_aware=True, require_offset=True),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--choralstream", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frame-head", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.choralstream).resolve()))
    from test import frame_metrics, load_gt_notes_from_pkl, load_model, notes_to_events, transcribe_one_song

    checkpoint = Path(args.checkpoint)
    model = load_model(checkpoint.parent.parent, checkpoint.name, args.device, 320)
    after = load_model(checkpoint.parent.parent, checkpoint.name, args.device, 320)
    head = torch.load(args.frame_head, map_location="cpu", weights_only=True)
    after.frame_prj.load_state_dict(head)
    unchanged = all(torch.equal(value, after.state_dict()[name]) for name, value in model.state_dict().items()
                    if not name.startswith("frame_prj."))
    changed_head = any(not torch.equal(value, after.state_dict()[name]) for name, value in model.state_dict().items()
                       if name.startswith("frame_prj."))
    if not unchanged or not changed_head:
        raise AssertionError("Expected only frame_prj parameters to change")

    ids = json.loads(Path(args.test_manifest).read_text())
    root = Path(args.data)
    totals = {key: (0, 0, 0) for key in (
        "note_onset_50ms", "note_onset_offset_50ms",
        "track_note_onset_50ms", "track_note_onset_offset_50ms",
        "frame_16ms", "track_frame_16ms",
    )}
    verified_one = False
    for index, fid in enumerate(ids):
        with torch.no_grad():
            pred = transcribe_one_song(model, root, fid, 320, "song", 0.5, 0.0, 0.08, 0.4, 2)
        if not verified_one:
            with torch.no_grad():
                after_pred = transcribe_one_song(after, root, fid, 320, "song", 0.5, 0.0, 0.08, 0.4, 2)
            verified_one = all(np.array_equal(pred[key], after_pred[key]) for key in ("pitch", "start", "dur", "voice"))
            if not verified_one:
                raise AssertionError("Event decoder changed after frame-head-only update")
        gt = load_gt_notes_from_pkl(root / "note" / f"{fid}.pkl")
        reference = notes_to_events(*gt)
        estimate = notes_to_events(pred["pitch"], pred["start"], pred["dur"], pred["voice"])
        for key, value in score_events(reference, estimate).items():
            totals[key] = add(totals[key], value)
        totals["frame_16ms"] = add(totals["frame_16ms"], frame_metrics(reference, estimate, track_aware=False))
        totals["track_frame_16ms"] = add(totals["track_frame_16ms"], frame_metrics(reference, estimate, track_aware=True))
        print(f"evaluated {index + 1}/{len(ids)}", flush=True)

    receipt = {
        "status": "measured_full_song_event_decoder_unchanged_by_frame_head_grpo",
        "songs": len(ids),
        "matched_test_manifest_sha256": hashlib.sha256(Path(args.test_manifest).read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "only_frame_head_parameters_changed": unchanged and changed_head,
        "paired_song_event_decode_identical": verified_one,
        "protocol": "Full-song beam=2, presence=0.5, 16ms frame grid, mir_eval note onset tolerance 50ms; offset tolerance max(50ms, 20% reference duration); pooled counts, both pitch-only and SATB-track-aware.",
        "before": {key: prf(value) for key, value in totals.items()},
        "after": {key: prf(value) for key, value in totals.items()},
        "caveat": "The post-trained frame head is not used by decode_events, so identical MIDI metrics are a structural consequence, not a null hypothesis test of event-level GRPO.",
    }
    Path(args.output).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
