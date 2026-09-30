"""Render beat times detected by the original Beat-v2 scorer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward-root", type=Path, required=True)
    parser.add_argument("--vocal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=12)
    parser.add_argument("--audio", action="append", required=True, help="LABEL=PATH")
    args = parser.parse_args()

    sys.path.insert(0, str(args.reward_root.resolve()))
    from mir.reward_function.beat_v2 import BeatV2Config, MadmomBeatV2Scorer

    scorer = MadmomBeatV2Scorer(BeatV2Config(segment_seconds=args.seconds, madmom_workers=1))
    reference = scorer._reference(args.vocal.resolve())
    rows = [{"label": "Vocal reference", "beats": reference.beats.tolist()}]
    for value in args.audio:
        label, path = value.split("=", 1)
        audio_path = Path(path).resolve()
        beats = scorer._active_times(scorer._detect_path(audio_path), reference.active, reference.sample_rate)
        result = scorer.score_paths(args.vocal, audio_path)
        rows.append({"label": label, "beats": beats.tolist(), "beat_v2_f1": result.score})

    fig, ax = plt.subplots(figsize=(11, 2.7), dpi=150)
    fig.patch.set_facecolor("#13191b")
    ax.set_facecolor("#13191b")
    colors = ["#68c4b5", "#aab7b8", "#e3b16d", "#e7776d"]
    labels = []
    for index, row in enumerate(rows):
        y = len(rows) - index
        color = colors[index % len(colors)]
        ax.hlines(y, 0, args.seconds, color="#344143", linewidth=1)
        ax.vlines(row["beats"], y - 0.25, y + 0.25, color=color, linewidth=2)
        f1 = row.get("beat_v2_f1")
        labels.append(row["label"] if f1 is None else f"{row['label']}  F1 {f1:.3f}")
    ax.set_yticks(list(range(len(rows), 0, -1)), labels)
    ax.set_xlim(0, args.seconds)
    ax.set_ylim(0.5, len(rows) + 0.5)
    ax.set_xlabel("Seconds", color="#bdcacc")
    ax.tick_params(colors="#bdcacc", labelsize=9, length=0)
    ax.spines[:].set_visible(False)
    ax.grid(axis="x", color="#344143", alpha=0.7)
    fig.tight_layout(pad=1.2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, facecolor=fig.get_facecolor())
    args.output.with_suffix(".json").write_text(json.dumps({"seconds": args.seconds, "rows": rows}, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
