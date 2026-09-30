"""Score a beat-aligned click track as a constructed Beat-v2 reward probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward-root", type=Path, required=True)
    parser.add_argument("--vocal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=6)
    args = parser.parse_args()

    sys.path.insert(0, str(args.reward_root.resolve()))
    from mir.reward_function.beat_v2 import BeatV2Config, MadmomBeatV2Scorer
    from mir.reward_function.coverage import accompaniment_coverage_path

    scorer = MadmomBeatV2Scorer(BeatV2Config(segment_seconds=args.seconds, madmom_workers=1))
    reference = scorer._reference(args.vocal.resolve()).beats
    sample_rate = 48000
    length = round(0.06 * sample_rate)
    t = np.arange(length) / sample_rate
    click = 0.55 * np.sin(2 * np.pi * 950 * t) * np.exp(-65 * t)

    def render(times: np.ndarray) -> np.ndarray:
        audio = np.zeros(round(args.seconds * sample_rate), dtype=np.float32)
        for beat in times:
            start = round(float(beat) * sample_rate)
            end = min(start + length, len(audio))
            if start < end:
                audio[start:end] += click[: end - start]
        return audio

    audio = render(reference)
    shifted = render(reference + 0.35)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, audio, sample_rate)
    shifted_output = args.output.with_name(args.output.stem + "_shifted.wav")
    sf.write(shifted_output, shifted, sample_rate)
    score = scorer.score_paths(args.vocal, args.output)
    shifted_score = scorer.score_paths(args.vocal, shifted_output)
    result = {
        "kind": "constructed_click_track_not_model_output",
        "seconds": args.seconds,
        "reference_beats": reference.tolist(),
        "beat_v2_f1": score.score,
        "detected_accompaniment_beats": score.accompaniment_beats,
        "stft_coverage": accompaniment_coverage_path(args.output),
        "rms": float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))),
        "peak": float(np.max(np.abs(audio))),
        "shifted_350ms_control": {
            "beat_v2_f1": shifted_score.score,
            "detected_accompaniment_beats": shifted_score.accompaniment_beats,
            "stft_coverage": accompaniment_coverage_path(shifted_output),
        },
    }
    receipt = args.output.with_suffix(".json")
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
