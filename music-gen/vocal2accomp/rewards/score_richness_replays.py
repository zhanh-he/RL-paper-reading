"""Offline richness-v0 diagnostic on existing matched LaDA replay audio."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer-dir", type=Path, required=True)
    parser.add_argument("--vocal", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.trainer_dir.resolve()))
    from train_lada_band import frame_rms, score_audio

    vocal, sr = sf.read(args.vocal, frames=12 * 48000, always_2d=True, dtype="float32")
    if sr != 48000:
        raise ValueError("Expected a 48 kHz Emma vocal")
    vocal_rms = frame_rms(vocal, sr)
    names = {
        "proxy": "grpo_emma_combined_6s",
        "coverage": "grpo_emma_coverage_6s",
        "beat_v2": "grpo_emma_beat_v2_6s",
        "guarded": "grpo_emma_guarded_6s",
        "beat_v5": "grpo_emma_beat_v5_6s",
        "richness_v0": "grpo_emma_richness_v0_6s",
    }
    rows = []
    for arm, directory in names.items():
        folder = args.runs_root / directory
        for path in sorted(folder.glob("step_*.wav")):
            audio, sample_rate = sf.read(path, always_2d=True, dtype="float32")
            if sample_rate != sr:
                raise ValueError(f"Unexpected sample rate: {path}")
            metrics = score_audio(audio, vocal_rms, sr, "richness_v0")
            row = {"arm": arm, "step": int(path.stem.split("_")[-1]), **metrics}
            rows.append(row)
            print(f"{arm} {row['step']}: richness-v0={row['reward']:.3f} layers={row['layer_activity']:.3f}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"protocol": "same Emma first 12 seconds; offline diagnostic only", "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
