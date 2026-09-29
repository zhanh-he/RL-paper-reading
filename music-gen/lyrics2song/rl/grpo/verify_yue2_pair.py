"""Check YuE2 untrained replay determinism and post-update token divergence."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    before = args.run / "heldout_before"
    replay = args.run / "heldout_replay"
    after = args.run / "heldout_after"
    tokens = [np.load(path / "semantic.npy") for path in (before, replay, after)]
    audio = [sf.read(path / "audio.wav", dtype="float32")[0] for path in (before, replay, after)]
    receipt = {
        "untrained_replay_semantic_identical": bool(np.array_equal(tokens[0], tokens[1])),
        "untrained_replay_audio_samples_identical": bool(np.array_equal(audio[0], audio[1])),
        "post_update_semantic_changed_positions": int(np.sum(tokens[0] != tokens[2])),
        "semantic_tokens": int(len(tokens[0])),
        "post_update_audio_rms_difference": float(np.sqrt(np.mean((audio[0].astype(np.float64) - audio[2]) ** 2))),
    }
    if not receipt["untrained_replay_audio_samples_identical"]:
        raise AssertionError("Baseline is not deterministic; do not attribute audio differences to GRPO")
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
