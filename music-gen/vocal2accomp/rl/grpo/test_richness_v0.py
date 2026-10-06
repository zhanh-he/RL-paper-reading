"""Constructed attacks for the experimental richness-v0 proxy."""

import unittest

import numpy as np

from train_lada_band import richness_v0_score, score_audio


class RichnessV0Test(unittest.TestCase):
    def test_constructed_signals(self):
        sr = 48000
        t = np.arange(6 * sr) / sr
        vocal_rms = np.full(150, 0.03)
        rng = np.random.default_rng(13)
        silence = np.zeros_like(t)
        sine = 0.03 * np.sin(2 * np.pi * 220 * t)
        noise = 0.03 * rng.standard_normal(len(t))
        clicks = np.zeros_like(t)
        for index in range(12):
            clicks[index * sr // 2:index * sr // 2 + 240] = 0.18
        layered = (
            0.022 * np.sin(2 * np.pi * 110 * t) * (0.3 + 0.7 * (np.sin(2 * np.pi * 2 * t) > 0))
            + 0.014 * np.sin(2 * np.pi * 440 * t) * (0.3 + 0.7 * (np.sin(2 * np.pi * t) > 0))
            + 0.009 * np.sin(2 * np.pi * 1760 * t) * (0.3 + 0.7 * (np.sin(2 * np.pi * 4 * t) > 0))
        )
        detuned = (
            0.022 * np.sin(2 * np.pi * 110 * t) * (0.3 + 0.7 * (np.sin(2 * np.pi * 2 * t) > 0))
            + 0.014 * np.sin(2 * np.pi * 463 * t) * (0.3 + 0.7 * (np.sin(2 * np.pi * t) > 0))
            + 0.009 * np.sin(2 * np.pi * 1760 * t) * (0.3 + 0.7 * (np.sin(2 * np.pi * 4 * t) > 0))
        )
        scores = {name: richness_v0_score(audio, vocal_rms, sr, 0)["reward"] for name, audio in {
            "silence": silence, "sine": sine, "noise": noise, "clicks": clicks,
            "layered": layered, "detuned": detuned,
        }.items()}
        for attack in ("silence", "sine", "noise", "clicks"):
            with self.subTest(attack=attack):
                self.assertGreater(scores["layered"], scores[attack], scores)
        self.assertLess(abs(scores["layered"] - scores["detuned"]), 0.03, scores)
        integrated = score_audio(np.repeat(layered[:, None], 2, axis=1), vocal_rms, sr, "richness_v0")
        self.assertIn("layer_activity", integrated)
        self.assertGreater(integrated["reward"], integrated["quality_penalty"])


if __name__ == "__main__":
    unittest.main()
