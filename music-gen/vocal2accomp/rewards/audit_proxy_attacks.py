"""Probe fast LaDA reward proxies with constructed non-musical signals."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocal", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=6)
    parser.add_argument("--trainer", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    trainer = args.trainer or Path(__file__).resolve().parents[1] / "rl/grpo/train_lada_band.py"
    sys.path.insert(0, str(trainer.resolve().parent))
    from train_lada_band import frame_rms, score_audio

    info = sf.info(args.vocal)
    source, sample_rate = sf.read(
        args.vocal,
        frames=round(args.seconds * info.samplerate),
        dtype="float32",
        always_2d=True,
    )
    if sample_rate != 48000:
        raise ValueError("the LaDA proxy audit expects 48 kHz vocal PCM")
    count = len(source)
    vocal_rms = frame_rms(source.mean(axis=1), sample_rate)
    time = np.arange(count) / sample_rate
    frame_time = (np.arange(len(vocal_rms)) + 0.5) * 0.04
    vocal_envelope = np.interp(time, frame_time, vocal_rms, left=vocal_rms[0], right=vocal_rms[-1])
    vocal_envelope /= max(float(vocal_envelope.max()), 1e-8)
    triad = (np.sin(2 * np.pi * 220 * time) + np.sin(2 * np.pi * 330 * time) + np.sin(2 * np.pi * 440 * time)) / 3
    signals = {
        "silence": np.zeros(count, dtype=np.float32),
        "sustained_220hz": (0.05 * np.sin(2 * np.pi * 220 * time)).astype(np.float32),
        "white_noise": np.random.default_rng(17).normal(0, 0.05, count).astype(np.float32),
        "vocal_gated_triad": ((0.03 + 0.09 * vocal_envelope) * triad).astype(np.float32),
    }
    results = {}
    for name, signal in signals.items():
        coverage = score_audio(signal, vocal_rms, sample_rate, "coverage")
        combined = score_audio(signal, vocal_rms, sample_rate, "combined")
        results[name] = {
            "coverage_only_reward": coverage["reward"],
            "combined_proxy_reward": combined["reward"],
            "rms_coverage": combined["rms_coverage"],
            "onset_fit_proxy": combined["onset_fit_proxy"],
            "band_occupancy_proxy": combined["band_occupancy_proxy"],
            "spectral_flatness": combined["spectral_flatness"],
            "quality_penalty": combined["quality_penalty"],
        }
    payload = {"kind": "constructed_reward_probes_not_model_outputs", "seconds": args.seconds, "results": results}
    serialized = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()
