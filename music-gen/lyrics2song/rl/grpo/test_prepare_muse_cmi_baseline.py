"""Run with: python3 -m unittest discover -s music-gen/lyrics2song/rl/grpo -p test_prepare_muse_cmi_baseline.py"""

import json
import tempfile
import unittest
from pathlib import Path

from prepare_muse_cmi_baseline import prepare


ROOT = Path(__file__).resolve().parents[4] / "datasets"


@unittest.skipUnless((ROOT / "cmi-pref-triple-text-lyrics-baseline-v1.json").exists(),
                     "Freeze projected split first")
class MuseProjectionTests(unittest.TestCase):
    def test_muse_prompts_use_frozen_membership_without_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = prepare(ROOT / "cmi-pref-triple-text-lyrics-baseline-v1.json",
                              Path(directory))
            self.assertEqual((summary["files"]["train"]["rows"],
                              summary["files"]["valid"]["rows"]), (240, 60))
            self.assertIs(summary["reference_audio_used"], False)
            self.assertEqual(summary["files"]["train"]["sha256"],
                             "37864c23487879ae21141270c0ebab046708d7f62ba66857c4c74b084c8969a8")
            for split in ("train", "valid"):
                path = Path(directory) / f"muse-cmi-triple-{split}.jsonl"
                for line in path.read_text().split("\n"):
                    if not line:
                        continue
                    messages = json.loads(line)["messages"]
                    self.assertEqual([message["role"] for message in messages],
                                     ["user", "assistant"])
                    self.assertIn("[lyrics:", messages[0]["content"])
                    self.assertNotIn("ref_audio", line)


if __name__ == "__main__":
    unittest.main()
