"""Fast checks for the LaDA-Band GRPO adapter and diagnostic rewards."""

import unittest

import numpy as np
import torch
from torch import nn

from train_lada_band import OutputLoRA, beat_coverage_guard_reward, frame_rms, score_audio


class RewardTests(unittest.TestCase):
    def test_coverage_and_quality_for_silence(self):
        quiet = np.zeros(48000, dtype=np.float32)
        result = score_audio(quiet, frame_rms(quiet, 48000), 48000, "combined")
        self.assertEqual(result["rms_coverage"], 0.0)
        self.assertEqual(result["clipping_fraction"], 0.0)
        self.assertTrue(np.isfinite(result["reward"]))

    def test_coverage_can_be_hacked_by_loud_constant_sound(self):
        vocal = np.zeros(48000, dtype=np.float32)
        loud = np.full(48000, 0.2, dtype=np.float32)
        result = score_audio(loud, frame_rms(vocal, 48000), 48000, "coverage")
        self.assertEqual(result["reward"], 1.0)
        self.assertEqual(result["onset_fit_proxy"], 0.0)

    def test_combined_penalizes_broadband_noise_despite_coverage(self):
        vocal = np.zeros(48000, dtype=np.float32)
        noise = np.random.default_rng(7).normal(0, 0.1, 48000).astype(np.float32)
        coverage = score_audio(noise, frame_rms(vocal, 48000), 48000, "coverage")
        combined = score_audio(noise, frame_rms(vocal, 48000), 48000, "combined")
        self.assertEqual(coverage["rms_coverage"], 1.0)
        self.assertGreater(combined["quality_penalty"], 0.0)
        self.assertLess(combined["reward"], coverage["reward"])

    def test_beat_coverage_guard_rejects_sparse_clicks_and_loudness(self):
        vocal_rms = np.full(150, 0.04, dtype=np.float32)
        quiet = np.full((48000, 2), 0.02, dtype=np.float32)
        loud = np.full((48000, 2), 0.08, dtype=np.float32)
        sparse = {"rms_coverage": 0.09, "spectral_flatness": 0.15, "quality_penalty": 0.0}
        active = {"rms_coverage": 0.7, "spectral_flatness": 0.15, "quality_penalty": 0.0}
        click = beat_coverage_guard_reward(1.0, sparse, quiet, vocal_rms)
        balanced = beat_coverage_guard_reward(0.5, active, quiet, vocal_rms)
        too_loud = beat_coverage_guard_reward(0.5, active, loud, vocal_rms)
        self.assertLess(click["reward"], 0.25)
        self.assertGreater(balanced["reward"], click["reward"])
        self.assertLess(too_loud["reward"], balanced["reward"])

    def test_beat_coverage_guard_penalizes_flat_noise(self):
        vocal_rms = np.full(150, 0.04, dtype=np.float32)
        audio = np.full((48000, 2), 0.02, dtype=np.float32)
        tonal = {"rms_coverage": 0.7, "spectral_flatness": 0.15, "quality_penalty": 0.0}
        noisy = {"rms_coverage": 0.7, "spectral_flatness": 0.85, "quality_penalty": 0.0}
        self.assertLess(
            beat_coverage_guard_reward(0.5, noisy, audio, vocal_rms)["reward"],
            beat_coverage_guard_reward(0.5, tonal, audio, vocal_rms)["reward"],
        )


class AdapterTests(unittest.TestCase):
    def test_zero_initialized_adapter_matches_base_and_receives_gradient(self):
        base = nn.Linear(8, 17, bias=False)
        adapter = OutputLoRA(base, rank=2, alpha=2)
        x = torch.randn(3, 8)
        torch.testing.assert_close(adapter(x), base(x))
        adapter(x).square().mean().backward()
        self.assertGreater(float(adapter.b.weight.grad.abs().sum()), 0)
        self.assertTrue(all(not parameter.requires_grad for parameter in base.parameters()))


if __name__ == "__main__":
    unittest.main()
