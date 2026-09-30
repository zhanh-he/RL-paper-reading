"""Verify LaDA training and held-out exports remain separate."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).with_name("publish-lada-demo.py")
SPEC = importlib.util.spec_from_file_location("publish_lada_demo", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PublisherTests(unittest.TestCase):
    def test_heldout_uses_its_own_vocal_and_checkpoint_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, heldout, site = (root / name for name in ("run", "heldout", "site"))
            for path in (run, heldout, site / "audio"):
                path.mkdir(parents=True)
            (run / "run.json").write_text(json.dumps({"args": {"reward": "combined", "seconds": 6, "eval_seconds": 12}}))
            (run / "evaluations.jsonl").write_text(json.dumps({"step": 0, "reward": 0.4}) + "\n")
            (run / "step_0000.wav").write_bytes(b"training accompaniment")
            (site / "audio/ace_emma_vocal_12s.wav").write_bytes(b"training vocal")

            (heldout / "source.wav").write_bytes(b"heldout vocal")
            (heldout / "step_0000.wav").write_bytes(b"heldout accompaniment")
            (heldout / "evaluations.json").write_text(json.dumps([
                {"step": 0, "duration_seconds": 6, "reward": 0.3},
            ]))

            with patch.object(MODULE.subprocess, "run"):
                training = MODULE.publish(run, site)
                heldout_result = MODULE.publish(run, site, heldout_dir=heldout)

            self.assertEqual(training.name, "vocal-lada-run.json")
            self.assertEqual(heldout_result.name, "vocal-lada-heldout-run.json")
            training_data = json.loads(training.read_text())
            heldout_data = json.loads(heldout_result.read_text())
            self.assertEqual(training_data["phrase"], "training_excerpt")
            self.assertEqual(heldout_data["phrase"], "heldout")
            self.assertEqual(heldout_data["config"]["eval_seconds"], 6)
            self.assertEqual((site / "audio/ace_emma_vocal_heldout_6s.wav").read_bytes(), b"heldout vocal")
            self.assertEqual((site / "audio/lada_emma_combined_heldout_step_0000.wav").read_bytes(), b"heldout accompaniment")
            self.assertEqual((site / "audio/lada_emma_combined_step_0000.wav").read_bytes(), b"training accompaniment")

    def test_heldout_beat_reward_uses_independent_madmom_score(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, heldout, site = (root / name for name in ("run", "heldout", "site"))
            for path in (run, heldout, site / "audio"):
                path.mkdir(parents=True)
            (run / "run.json").write_text(json.dumps({"args": {"reward": "beat_v2", "seconds": 6, "eval_seconds": 12}}))
            (heldout / "source.wav").write_bytes(b"heldout vocal")
            (heldout / "step_0000.wav").write_bytes(b"heldout accompaniment")
            (heldout / "evaluations.json").write_text(json.dumps([{"step": 0, "duration_seconds": 6, "reward": None, "proxy_reward": 0.8}]))
            (heldout / "beat_v2.json").write_text(json.dumps({"0": {"score": 0.25}}))

            with patch.object(MODULE.subprocess, "run"):
                output = MODULE.publish(run, site, slug="beat_v2", heldout_dir=heldout)

            self.assertEqual(output.name, "vocal-lada-beat_v2-heldout-run.json")
            data = json.loads(output.read_text())
            self.assertEqual(data["stages"][0]["reward"], 0.25)
            self.assertEqual(data["stages"][0]["beat_v2"]["score"], 0.25)
            self.assertEqual((site / "audio/lada_emma_beat_v2_heldout_step_0000.wav").read_bytes(), b"heldout accompaniment")

    def test_guarded_training_arm_has_its_own_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, site = root / "run", root / "site"
            run.mkdir()
            (site / "audio").mkdir(parents=True)
            (site / "audio/ace_emma_vocal_12s.wav").write_bytes(b"vocal")
            (run / "run.json").write_text(json.dumps({"args": {"reward": "beat_v2_coverage_guard"}}))
            (run / "evaluations.jsonl").write_text(json.dumps({"step": 0, "reward": 0.2}) + "\n")
            (run / "step_0000.wav").write_bytes(b"accompaniment")

            with patch.object(MODULE.subprocess, "run"):
                output = MODULE.publish(run, site, slug="guarded")

            self.assertEqual(output.name, "vocal-lada-guarded-run.json")
            self.assertEqual(json.loads(output.read_text())["stages"][0]["reward"], 0.2)

    def test_heldout_guarded_reward_is_recomputed_from_original_beat_score(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, heldout, site = (root / name for name in ("run", "heldout", "site"))
            for path in (run, heldout, site / "audio"):
                path.mkdir(parents=True)
            (run / "run.json").write_text(json.dumps({"args": {"reward": "beat_v2_coverage_guard"}}))
            (heldout / "source.wav").write_bytes(b"vocal")
            (heldout / "step_0000.wav").write_bytes(b"accompaniment")
            (heldout / "evaluations.json").write_text(json.dumps([{
                "step": 0, "duration_seconds": 6, "reward": None, "proxy_reward": 0.8,
                "rms_coverage": 0.8, "quality_penalty": 0.0, "spectral_flatness": 0.2,
            }]))
            (heldout / "beat_v2.json").write_text(json.dumps({"0": {"score": 0.5, "acc_to_vocal_rms_db": -5}}))

            with patch.object(MODULE.subprocess, "run"):
                output = MODULE.publish(run, site, slug="guarded", heldout_dir=heldout)

            data = json.loads(output.read_text())
            self.assertAlmostEqual(data["stages"][0]["reward"], 0.675)
            self.assertNotEqual(data["stages"][0]["reward"], 0.8)


if __name__ == "__main__":
    unittest.main()
