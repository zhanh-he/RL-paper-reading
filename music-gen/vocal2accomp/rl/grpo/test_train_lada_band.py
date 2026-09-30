"""Fast checks for the LaDA-Band GRPO adapter and diagnostic rewards."""

import unittest

import numpy as np
import torch
from torch import nn

from train_lada_band import OutputLoRA, frame_rms, score_audio


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
