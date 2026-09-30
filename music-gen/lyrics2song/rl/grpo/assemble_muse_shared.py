"""Bind the shared Muse replay audio to freshly computed dual-reward scores."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[4]
RUNS = Path(__file__).resolve().parent / "runs"
COMMON = RUNS / "2026-09-30-muse-shared-replay"
DEMOS = ROOT / "platform" / "site" / "demos"
DIMENSIONS = ("Coherence", "Musicality", "Memorability", "Clarity", "Naturalness")
AUDIO = {
    "baseline": "muse_songeval_matched_before.flac",
    "songeval-1": "muse_songeval_matched_after.flac",
    "musecritic-1": "muse_shared_musecritic_000001.flac",
    "musecritic-25": "muse_shared_musecritic_000025.flac",
    "musecritic-50": "muse_shared_musecritic_000050.flac",
}


def signal(path: Path) -> dict:
    samples, sample_rate = sf.read(path, dtype="float32")
    amplitude = np.abs(samples)
    peak = float(amplitude.max())
    return {
        "seconds": len(samples) / sample_rate,
        "sample_rate": sample_rate,
        "peak": peak,
        "rms": float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))),
        "near_full_scale_fraction": float(np.mean(amplitude >= 0.999)),
        "flat_top_fraction": float(np.mean(np.isclose(amplitude, peak, atol=1e-6))),
    }


def reward(scores: dict) -> dict:
    return {**{key: float(scores[key]) for key in DIMENSIONS},
            "mean": sum(float(scores[key]) for key in DIMENSIONS) / len(DIMENSIONS)}


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    tokens = json.loads((COMMON / "token-receipt.json").read_text())
    songeval = json.loads((COMMON / "songeval_scores.json").read_text())
    musecritic = json.loads((COMMON / "musecritic_scores.json").read_text())
    rows = {}
    for name, filename in AUDIO.items():
        audio_path = DEMOS / "audio" / filename
        song_score = reward(songeval[f"{name}_tokens"])
        critic_score = reward(musecritic[name])
        metrics = signal(audio_path)
        if metrics["sample_rate"] != 48000 or abs(metrics["seconds"] - 19.76) > 0.1:
            raise ValueError(f"Unexpected shared replay format: {audio_path}")
        rows[name] = {
            "audio": f"audio/{filename}",
            "audio_sha256": hashlib.sha256(audio_path.read_bytes()).hexdigest(),
            "token_stage": tokens["stages"][name],
            "signal": metrics,
            "songeval": song_score,
            "musecritic": critic_score,
        }
    result = {
        "status": "measured_shared_muse_replay",
        "model": "Muse-0.6b + MuCodec",
        "prompt": tokens["prompt"],
        "seed": tokens["seed"],
        "max_new_tokens": tokens["max_new_tokens"],
        "force_length": tokens["force_length"],
        "sampling": tokens["sampling"],
        "mucodec_steps": 20,
        "rows": rows,
        "caveat": "One prompt and seed; the two reward scales are separate and these scores are not human preference judgments.",
    }
    write(COMMON / "receipt.json", result)
    for reward_name, stages in (("songeval", (1,)), ("musecritic", (1, 25, 50))):
        for step in stages:
            after = rows[f"{reward_name}-{step}"]
            before = rows["baseline"]
            write(RUNS / f"2026-09-30-muse-shared-{reward_name}" / f"step_{step:06d}" / "receipt.json", {
                "status": "measured_shared_prompt_muse_pair",
                "model": result["model"],
                "reward_model": "SongEval" if reward_name == "songeval" else "MuseCritic",
                "optimizer_step": step,
                "heldout_prompt": result["prompt"],
                "heldout_seed": result["seed"],
                "shared_replay_receipt": "music-gen/lyrics2song/rl/grpo/runs/2026-09-30-muse-shared-replay/receipt.json",
                "before": {"reward": before[reward_name], "signal": before["signal"], "audio_sha256": before["audio_sha256"]},
                "after": {"reward": after[reward_name], "signal": after["signal"], "audio_sha256": after["audio_sha256"]},
                "cross_reward": {
                    "before": before["musecritic" if reward_name == "songeval" else "songeval"],
                    "after": after["musecritic" if reward_name == "songeval" else "songeval"],
                },
                "caveat": result["caveat"],
            })
    print("Wrote shared Muse replay receipts")


if __name__ == "__main__":
    main()
