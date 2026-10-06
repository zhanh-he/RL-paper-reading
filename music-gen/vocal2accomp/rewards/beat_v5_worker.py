"""Persistent Beat-v5 scorer for matched LaDA online reward rollouts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward-root", type=Path, required=True)
    parser.add_argument("--vocal", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.reward_root.resolve()))
    from mir.reward_function.beat_v5 import BeatV5Config, BeatV5Scorer

    scorers = {}
    for line in sys.stdin:
        try:
            request = json.loads(line)
            seconds = float(request["seconds"])
            scorer = scorers.get(seconds)
            if scorer is None:
                scorer = BeatV5Scorer(
                    BeatV5Config(
                        segment_seconds=seconds,
                        window_seconds=min(12.0, seconds),
                        window_hop_seconds=min(4.0, seconds / 2),
                        madmom_workers=1,
                    ),
                    backend="madmom",
                )
                scorers[seconds] = scorer
            result = scorer.score_paths(args.vocal, request["audio"])
            response = {
                "score": result.score,
                "scorable": not result.abstain,
                "confidence": result.confidence,
                "reasons": result.reasons,
                "reference_beats": result.diagnostics["vocal_onsets"],
                "accompaniment_beats": result.diagnostics["accompaniment_beats"],
            }
        except Exception as exc:
            response = {"error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
