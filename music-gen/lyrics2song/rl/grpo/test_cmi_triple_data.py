"""Run with: python3 -m unittest discover -s music-gen/lyrics2song/rl/grpo -p test_cmi_triple_data.py"""

import json
import unittest
from pathlib import Path

from prepare_cmi_triple_data import prepare
from prepare_formal_data import normalized, rows, sha256


ROOT = Path(__file__).resolve().parents[4] / "datasets"


@unittest.skipUnless((ROOT / "cmi-pref/cmi_train.jsonl").exists(), "Download pinned metadata first")
class TripleDataTests(unittest.TestCase):
    def test_nine_to_one_public_split_matches_private_manifest(self):
        public_path = Path(__file__).resolve().parents[4] / "platform/site/demos/data/cmi-pref-triple-9to1-split.json"
        summary = prepare(ROOT, validation_conditions=30,
                          manifest_name="cmi-pref-triple-9to1-v2.json",
                          public_split_path=public_path)
        manifest = json.loads((ROOT / "cmi-pref-triple-9to1-v2.json").read_text())
        public = json.loads(public_path.read_text())
        self.assertEqual((len(public["train"]), len(public["valid"]), len(public["test"])), (270, 30, 121))
        self.assertEqual(summary["manifest_sha256"], public["private_manifest_sha256"])
        for split in ("train", "valid"):
            self.assertEqual({row["condition_id"] for row in manifest[split]},
                             {row["condition_id"] for row in public[split]})
        ids = [{row["condition_id"] for row in public[split]} for split in ("train", "valid", "test")]
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])

    def test_reproducible_and_disjoint(self):
        summary = prepare(ROOT)
        path = ROOT / "cmi-pref-triple-v1.json"
        manifest = json.loads(path.read_text())
        self.assertEqual(sha256(path), summary["manifest_sha256"])
        self.assertEqual(summary["manifest_sha256"],
                         "b611333ef0abdeaf7a0473ea0beaa470673c3414f05b816a97904d2ade25b2e0")
        self.assertEqual((len(manifest["train"]), len(manifest["valid"])), (240, 60))
        test = rows(ROOT / "cmi-pref/cmi_test.jsonl")
        wsb = rows(ROOT / "wildsongbench/reproduction_manifest.jsonl")
        sealed_lyrics = {normalized(row["lyrics"]) for row in test + wsb if row["lyrics"].strip()}
        train_lyrics = {normalized(row["lyrics"]) for row in manifest["train"]}
        valid_lyrics = {normalized(row["lyrics"]) for row in manifest["valid"]}
        self.assertFalse(train_lyrics & valid_lyrics)
        self.assertFalse((train_lyrics | valid_lyrics) & sealed_lyrics)
        test_refs = {row["ref-audio-path"] for row in test if row["ref-audio-path"].strip()}
        train_refs = {row["ref_audio_path"] for row in manifest["train"]}
        valid_refs = {row["ref_audio_path"] for row in manifest["valid"]}
        self.assertFalse(train_refs & valid_refs)
        self.assertFalse((train_refs | valid_refs) & test_refs)
        self.assertEqual(summary["counts"]["exact_official_train_test_condition_overlap"], 22)
        self.assertEqual(summary["counts"]["decontaminated_train_votes"], 435)


if __name__ == "__main__":
    unittest.main()
