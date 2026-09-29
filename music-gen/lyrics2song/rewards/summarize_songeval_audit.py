"""Publish only paired aggregate effects from a private SongEval perturbation audit."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    receipt = json.loads(args.input.read_text())
    rows = receipt["rows"]
    clips = receipt["clips"]
    labels = (*rows[0]["scores"], "five_mean")
    index = {(row["clip_index"], row["variant"]): row for row in rows}
    rng = np.random.default_rng(20260929)
    pairs = (
        ("safe_gain", "reference"),
        ("hard_clip_full_scale", "reference"),
        ("hard_clip_rms_matched", "safe_gain"),
        ("bandlimit_6khz_upsampled", "reference"),
    )
    comparisons = {}
    def value(row, label):
        return float(np.mean(list(row["scores"].values()))) if label == "five_mean" else row["scores"][label]

    for treatment, control in pairs:
        key = f"{treatment}_minus_{control}"
        comparisons[key] = {}
        for label in labels:
            delta = np.array([value(index[i, treatment], label) - value(index[i, control], label)
                              for i in range(clips)])
            bootstrap = rng.choice(delta, size=(10000, clips), replace=True).mean(axis=1)
            comparisons[key][label] = {
                "mean": float(delta.mean()), "median": float(np.median(delta)),
                "positive": int(np.sum(delta > 0)), "n": clips,
                "paired_bootstrap_95pct": [float(x) for x in np.quantile(bootstrap, [.025, .975])],
            }
    signal = {}
    for variant in {row["variant"] for row in rows}:
        signal[variant] = {
            key: float(np.mean([index[i, variant]["signal"][key] for i in range(clips)]))
            for key in ("peak", "rms", "flat_top_fraction")
        }
    public = {
        "status": "controlled_perturbation_audit_not_grpo",
        "date": "2026-09-29",
        "clips": clips,
        "reward": "Official SongEval checkpoint; five direct output scores without a training-specific reward wrapper",
        "input": "evaluation-only music excerpts; no audio, source IDs or individual scores published",
        "method": "Peak-normalize each original mono waveform to 0.5; safe gain x1.6; hard clip after x3.2; separately RMS-match hard clip to safe gain; 6kHz lowpass/upsample control. Score through SongEval 24kHz mono path.",
        "comparisons": comparisons,
        "mean_signal": signal,
        "limitations": "Controlled waveform perturbations do not establish GRPO behavior or a causal explanation for any proprietary model. Bootstrap resamples clips, not model seeds.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(public, indent=2) + "\n")
    print(json.dumps(public, indent=2))


if __name__ == "__main__":
    main()
