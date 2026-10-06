"""Score matched fixed LaDA replays with original Beat-v2 and Beat-v5."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward-root", type=Path, required=True)
    parser.add_argument("--vocal", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.reward_root.resolve()))
    from mir.reward_function.beat_v2 import BeatV2Config, MadmomBeatV2Scorer
    from mir.reward_function.beat_v5 import BeatV5Config, BeatV5Scorer

    v2 = MadmomBeatV2Scorer(BeatV2Config(segment_seconds=12, madmom_workers=1))
    v5 = BeatV5Scorer(BeatV5Config(segment_seconds=12, madmom_workers=1), backend="madmom")
    runs = {
        "proxy": "grpo_emma_combined_6s",
        "coverage": "grpo_emma_coverage_6s",
        "beat_v2": "grpo_emma_beat_v2_6s",
        "guarded": "grpo_emma_guarded_6s",
        "beat_v5": "grpo_emma_beat_v5_6s",
        "richness_v0": "grpo_emma_richness_v0_6s",
    }
    rows = []
    for arm, directory in runs.items():
        log = args.runs_root / directory / "evaluations.jsonl"
        if not log.is_file():
            continue
        for line in log.read_text().splitlines():
            item = json.loads(line)
            path = Path(item["audio"])
            if not path.is_file():
                continue
            score2 = v2.score_paths(args.vocal, path)
            score5 = v5.score_paths(args.vocal, path)
            rows.append({
                "arm": arm,
                "step": item["step"],
                "beat_v2_f1": score2.score,
                "beat_v2_scorable": score2.scorable,
                "beat_v2_reference_beats": score2.reference_beats,
                "beat_v2_accompaniment_beats": score2.accompaniment_beats,
                "beat_v5_score": score5.score,
                "beat_v5_confidence": score5.confidence,
                "beat_v5_abstain": score5.abstain,
                "beat_v5_reasons": list(score5.reasons),
                "beat_v5_components": dict(score5.components),
                "beat_v5_diagnostics": dict(score5.diagnostics),
            })
            print(f"{arm} {item['step']}: v2={score2.score:.3f} v5={score5.score:.3f} confidence={score5.confidence:.3f}", flush=True)
    result = {
        "protocol": "same original Emma vocal file, first 12 seconds scored; same fixed replays; original v2 and madmom-backend v5",
        "vocal_sha256": hashlib.sha256(args.vocal.read_bytes()).hexdigest(),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
