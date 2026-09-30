"""Score fixed Muse replays with the official MuseCritic deployment model."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch


NAMES = ("baseline", "songeval-1", "musecritic-1", "musecritic-25", "musecritic-50")
SCORE_KEYS = ("Coherence", "Musicality", "Memorability", "Clarity", "Naturalness")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--musecritic", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.musecritic / "muse_grpo"))
    from plugin import MUSECRITIC_USER_TEXT, _musecritic_prompt_with_single_audio
    sys.path.insert(0, str(args.musecritic / "muse_grpo" / "deploy" / "musecritic"))
    from musecritic_serve import MuseCriticServer

    prompt = re.sub(r"\n?<audio>\n?", "", _musecritic_prompt_with_single_audio(MUSECRITIC_USER_TEXT), flags=re.I)
    server = MuseCriticServer(args.model, torch.device("cuda:0"), default_max_tokens=512)
    server.load_model()
    results = {}
    for name in NAMES:
        audio = args.replay / f"{name}_tokens.wav"
        if not audio.is_file():
            raise FileNotFoundError(audio)
        detail = server.predict(prompt, str(audio), max_new_tokens=512)
        scores = detail["reward_scores"]
        results[name] = {**scores, "mean": sum(float(scores[key]) for key in SCORE_KEYS) / len(SCORE_KEYS)}
        print(f"{name}: MuseCritic {results[name]['mean']:.4f}", flush=True)
    (args.replay / "musecritic_scores.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
