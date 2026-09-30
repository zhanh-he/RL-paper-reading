"""Persistent original Beat-v2 scorer for LaDA online reward rollouts."""

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
    from mir.reward_function.beat_v2 import BeatV2Config, MadmomBeatV2Scorer

    scorers = {}
    for line in sys.stdin:
        try:
            request = json.loads(line)
            seconds = float(request["seconds"])
            scorer = scorers.get(seconds)
            if scorer is None:
                scorer = MadmomBeatV2Scorer(BeatV2Config(segment_seconds=seconds, madmom_workers=1))
                scorers[seconds] = scorer
            result = scorer.score_paths(args.vocal, request["audio"])
            response = {
                "score": result.score,
                "scorable": result.scorable,
                "reference_beats": result.reference_beats,
                "accompaniment_beats": result.accompaniment_beats,
            }
        except Exception as exc:
            response = {"error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
