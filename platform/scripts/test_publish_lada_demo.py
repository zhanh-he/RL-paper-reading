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


if __name__ == "__main__":
    unittest.main()
