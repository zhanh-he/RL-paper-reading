"""BCR-v1: explicit Beat-v5 / STFT Coverage / Richness-v0 objectives.

The layer term remains a spectral proxy. Weights are fixed before training;
raw components and shared guardrails must be logged for every experiment arm.
"""
from __future__ import annotations

import math

WEIGHTS = {
    "bcr": (0.50, 0.25, 0.25),
    "no_beat": (0.00, 0.50, 0.50),
    "no_coverage": (0.67, 0.00, 0.33),
    "no_richness": (0.67, 0.33, 0.00),
    "beat": (1.00, 0.00, 0.00),
    "coverage": (0.00, 1.00, 0.00),
    "richness": (0.00, 0.00, 1.00),
}


def combine(beat: float, coverage: float, richness: float, *, arm: str = "bcr",
            quality_penalty: float = 0.0, loudness_penalty: float = 0.0,
            beat_scorable: bool = True) -> dict:
    values = (beat, coverage, richness, quality_penalty, loudness_penalty)
    if not all(math.isfinite(x) for x in values):
        raise ValueError("Nonfinite reward component")
    if any(not 0 <= x <= 1 for x in (beat, coverage, richness)):
        raise ValueError("Raw rewards must lie in [0, 1]")
    if quality_penalty < 0 or loudness_penalty < 0:
        raise ValueError("Penalties must be nonnegative")
    weights = WEIGHTS[arm]
    # The coverage cap stops incentivizing additional energy after enough activity.
    c = min(1.0, coverage / 0.7)
    b = beat if beat_scorable else 0.0
    reward = weights[0] * b + weights[1] * c + weights[2] * richness
    return {"reward": reward - quality_penalty - loudness_penalty,
            "beat_v5": b, "beat_v5_raw": beat, "beat_scorable": beat_scorable,
            "stft_coverage": coverage, "coverage_capped": c,
            "richness_v0_raw": richness, "quality_penalty": quality_penalty,
            "reference_loudness_penalty": loudness_penalty,
            "reward_version": "bcr-v1", "reward_weights": list(weights)}
