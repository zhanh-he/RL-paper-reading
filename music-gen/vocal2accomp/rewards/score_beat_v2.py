"""Score fixed-vocal LaDA checkpoints with the existing Madmom Beat-v2 implementation."""

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
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=12)
    args = parser.parse_args()

    sys.path.insert(0, str(args.reward_root.resolve()))
    from mir.reward_function.beat_v2 import BeatV2Config, MadmomBeatV2Scorer
    from mir.reward_function.coverage import accompaniment_coverage_path

    scorer = MadmomBeatV2Scorer(BeatV2Config(segment_seconds=args.seconds, madmom_workers=1))
    vocal_info = sf.info(args.vocal)
    vocal_waveform, _ = sf.read(args.vocal, frames=round(args.seconds * vocal_info.samplerate), dtype="float32", always_2d=True)
    vocal_mono = vocal_waveform.mean(axis=1)
    vocal_rms = float(np.sqrt(np.mean(np.square(vocal_mono, dtype=np.float64))))
    output = {}
    for audio in sorted(args.run_dir.glob("step_*.wav")):
        step = int(audio.stem.split("_")[-1])
        result = scorer.score_paths(args.vocal, audio)
        waveform, _ = sf.read(audio, dtype="float32", always_2d=True)
        stereo_rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))
        output[str(step)] = {
            "score": result.score,
            "reference_beats": result.reference_beats,
            "accompaniment_beats": result.accompaniment_beats,
            "scorable": result.scorable,
            "coverage_stft": accompaniment_coverage_path(audio),
            "stereo_rms": stereo_rms,
            "vocal_rms": vocal_rms,
            "acc_to_vocal_rms_db": float(20 * np.log10(max(stereo_rms, 1e-12) / max(vocal_rms, 1e-12))),
            "stereo_peak": float(np.max(np.abs(waveform))),
            "stereo_clipping_fraction": float(np.mean(np.abs(waveform) >= 0.98)),
        }
    target = args.run_dir / "beat_v2.json"
    target.write_text(json.dumps(output, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
