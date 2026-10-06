"""Run with: python3 -m unittest discover -s music-gen/lyrics2song/rl/grpo -p test_project_cmi_triple_baseline.py"""

import json
import unittest
from pathlib import Path

from project_cmi_triple_baseline import SOURCE_SHA256, project
from prepare_formal_data import normalized, sha256


ROOT = Path(__file__).resolve().parents[4] / "datasets"


@unittest.skipUnless((ROOT / "cmi-pref-triple-v1.json").exists(), "Freeze three-part source first")
class ProjectionTests(unittest.TestCase):
    def test_nine_to_one_projection(self):
        source_path = ROOT / "cmi-pref-triple-9to1-v2.json"
        target_path = ROOT / "cmi-pref-triple-text-lyrics-9to1-v2.json"
        self.assertEqual(project(source_path, target_path, 30),
                         "c39d43e81793bee3f312c7028a0483b5da41068db4544acd258115043101e46f")
        target = json.loads(target_path.read_text())
        self.assertEqual((len(target["train"]), len(target["valid"])), (270, 30))
        self.assertFalse(any("audio" in row for row in target["train"] + target["valid"]))

    def test_projection_preserves_membership_and_drops_audio(self):
        source_path = ROOT / "cmi-pref-triple-v1.json"
        target_path = ROOT / "cmi-pref-triple-text-lyrics-baseline-v1.json"
        self.assertEqual(sha256(source_path), SOURCE_SHA256)
        projected_sha = project(source_path, target_path)
        self.assertEqual(projected_sha,
                         "b061588397d54177b25b678962caf756771498b927b9931542908c5e64a7e109")
        source = json.loads(source_path.read_text())
        target = json.loads(target_path.read_text())
        self.assertEqual(target["input_modalities"], ["text", "lyrics"])
        self.assertIs(target["reference_audio_used"], False)
        for split, expected in (("train", 240), ("valid", 60)):
            self.assertEqual(len(target[split]), expected)
            self.assertEqual([item["condition_id"] for item in target[split]],
                             [item["condition_id"] for item in source[split]])
            self.assertTrue(all(set(item) == {"condition_id", "style", "lyrics"}
                                for item in target[split]))
        train_lyrics = {normalized(item["lyrics"]) for item in target["train"]}
        valid_lyrics = {normalized(item["lyrics"]) for item in target["valid"]}
        self.assertFalse(train_lyrics & valid_lyrics)


if __name__ == "__main__":
    unittest.main()
