"""Run with: python3 -m unittest discover -s music-gen/lyrics2song/rl/grpo -p test_formal_data.py"""

import json
import unittest
from pathlib import Path

from prepare_formal_data import normalized, prepare, rows, sha256


ROOT = Path(__file__).resolve().parents[4] / "datasets"


@unittest.skipUnless((ROOT / "formal-lyrics2song-v1.json").exists(), "Download pinned metadata first")
class FormalDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest_path = ROOT / "formal-lyrics2song-v1.json"
        cls.manifest = json.loads(cls.manifest_path.read_text())

    def test_reproducible_manifest(self):
        before = sha256(self.manifest_path)
        summary = prepare(ROOT)
        self.assertEqual(before, sha256(self.manifest_path))
        self.assertEqual(before, summary["manifest_sha256"])

    def test_no_lyrics_leakage(self):
        train = self.manifest["train"]
        valid = self.manifest["valid"]
        self.assertEqual((len(train), len(valid)), (234, 59))
        test = rows(ROOT / "cmi-pref/cmi_test.jsonl")
        wsb = rows(ROOT / "wildsongbench/reproduction_manifest.jsonl")
        sealed = {normalized(row["lyrics"]) for row in test + wsb if row["lyrics"].strip()}
        train_lyrics = {normalized(row["lyrics"]) for row in train}
        valid_lyrics = {normalized(row["lyrics"]) for row in valid}
        self.assertFalse((train_lyrics | valid_lyrics) & sealed)
        self.assertFalse(train_lyrics & valid_lyrics)
        self.assertEqual(len({row["condition_id"] for row in train + valid}), 293)


if __name__ == "__main__":
    unittest.main()
