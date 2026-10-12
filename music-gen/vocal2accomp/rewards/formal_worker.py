"""Persistent multi-song original Beat-v5 / STFT Coverage worker."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward-root", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.reward_root.resolve()))
    from mir.reward_function.beat_v5 import BeatV5Config, BeatV5Scorer
    from mir.reward_function.coverage import accompaniment_coverage_path
    BeatV5Scorer._compatibility_shims()
    scorers = {}
    for line in sys.stdin:
        try:
            request = json.loads(line)
            seconds = float(request["seconds"])
            if seconds not in scorers:
                scorers[seconds] = BeatV5Scorer(BeatV5Config(
                    segment_seconds=seconds, window_seconds=min(12, seconds),
                    window_hop_seconds=min(4, seconds / 2), madmom_workers=1), backend="madmom")
            result = scorers[seconds].score_paths(request["vocal"], request["audio"])
            response = {"score": result.score, "scorable": not result.abstain,
                        "confidence": result.confidence, "reasons": result.reasons,
                        "components": dict(result.components),
                        "reference_beats": result.diagnostics["vocal_onsets"],
                        "accompaniment_beats": result.diagnostics["accompaniment_beats"],
                        "stft_coverage": accompaniment_coverage_path(request["audio"])}
        except Exception as exc:
            response = {"error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
