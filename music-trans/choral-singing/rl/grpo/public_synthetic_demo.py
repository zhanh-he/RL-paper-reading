"""Render an original SATB example and replay ChoralStream before/after frame GRPO.

The original harmonic synthesis is an illustration, not a held-out benchmark.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


PITCHES = (
    (72, 74, 76, 74, 72, 71, 69, 72),
    (67, 69, 69, 67, 65, 64, 65, 67),
    (60, 62, 64, 62, 60, 59, 57, 60),
    (48, 50, 52, 55, 53, 52, 50, 48),
)
SAMPLE_RATE = 16000


def reference_notes():
    return [
        {"voice": voice, "pitch": pitch, "start": index * 1.25,
         "end": index * 1.25 + 1.12}
        for voice, line in enumerate(PITCHES)
        for index, pitch in enumerate(line)
    ]


def render(notes, seconds=10.2):
    audio = np.zeros(round(seconds * SAMPLE_RATE), dtype=np.float64)
    for note in notes:
        start = max(0, round(note["start"] * SAMPLE_RATE))
        end = min(len(audio), round(note["end"] * SAMPLE_RATE))
        if end <= start:
            continue
        t = np.arange(end - start) / SAMPLE_RATE
        frequency = 440 * 2 ** ((note["pitch"] - 69) / 12)
        tone = sum(
            np.sin(2 * np.pi * frequency * harmonic * t) * weight
            for harmonic, weight in enumerate((0.7, 0.25, 0.12, 0.06), start=1)
        )
        envelope = np.minimum(1, t / 0.055) * np.minimum(1, (t[-1] - t) / 0.11)
        audio[start:end] += tone * envelope * (0.15 if note["voice"] < 2 else 0.20)
    peak = np.max(np.abs(audio))
    return (audio / max(peak / 0.85, 1)).astype(np.float32)


def arrays(notes):
    return tuple(np.asarray([note[key] for note in notes]) for key in ("pitch", "start", "voice"))


def save_midi(notes, path, notes_to_midi):
    pitch, start, voice = arrays(notes)
    duration = np.asarray([note["end"] - note["start"] for note in notes])
    notes_to_midi(pitch, start, duration, voice, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--choralstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--frame-head", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.choralstream.resolve()))
    from test import frame_metrics, load_model
    from transcribe import load_mel, notes_to_midi, run_transcription
    from evaluate_notes import prf, score_events

    reference = reference_notes()
    sf.write(args.output / "input.wav", render(reference), SAMPLE_RATE)
    sf.write(args.output / "reference.wav", render(reference), SAMPLE_RATE)
    save_midi(reference, args.output / "reference.mid", notes_to_midi)
    mel, _ = load_mel(input_audio=str(args.output / "input.wav"))
    base = load_model(args.checkpoint.parent.parent, args.checkpoint.name, args.device, 320)
    trained = load_model(args.checkpoint.parent.parent, args.checkpoint.name, args.device, 320)
    trained.frame_prj.load_state_dict(torch.load(args.frame_head, map_location="cpu", weights_only=True))

    predictions = {}
    for name, model in (("baseline", base), ("grpo", trained)):
        with torch.no_grad():
            raw = run_transcription(model, mel, 320, 0.5, 0.0, 0.08, 0.4, 2,
                                    "song", audio_path=args.output / "input.wav")
        predictions[name] = [
            {"voice": int(v), "pitch": int(p), "start": float(s), "end": float(s + d)}
            for p, s, d, v in zip(raw["pitch"], raw["start"], raw["dur"], raw["voice"])
        ]
        save_midi(predictions[name], args.output / f"{name}.mid", notes_to_midi)
        sf.write(args.output / f"{name}.wav", render(predictions[name]), SAMPLE_RATE)
        print(f"{name}: {len(predictions[name])} notes", flush=True)

    metrics = {}
    for name, notes in predictions.items():
        scored = {key: prf(value) for key, value in score_events(reference, notes).items()}
        scored["frame_16ms"] = prf(frame_metrics(reference, notes, track_aware=False))
        scored["track_frame_16ms"] = prf(frame_metrics(reference, notes, track_aware=True))
        metrics[name] = scored
    receipt = {
        "status": "public_synthetic_illustration_not_heldout_benchmark",
        "scope": "Original harmonic SATB synthesis; the trained checkpoint changes frame_prj only",
        "reference_notes": len(reference),
        "predicted_notes": {name: len(notes) for name, notes in predictions.items()},
        "event_decode_identical": predictions["baseline"] == predictions["grpo"],
        "protocol": "mir_eval onset tolerance 50ms; offset tolerance max(50ms, 20% reference duration)",
        "metrics": metrics,
    }
    (args.output / "notes.json").write_text(json.dumps({"reference": reference, **predictions}, indent=2) + "\n")
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
