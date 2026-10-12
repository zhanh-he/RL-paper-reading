"""Small deterministic checks before dispatching expensive multi-song jobs."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rewards"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from combine import combine


class CombinationTests(unittest.TestCase):
    def test_all_three_components_contribute(self):
        empty = combine(0, 0, 0)["reward"]
        for scores in ((0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)):
            self.assertGreater(combine(*scores)["reward"], empty)

    def test_removed_component_cannot_affect_arm(self):
        for arm, index in (("no_beat", 0), ("no_coverage", 1), ("no_richness", 2)):
            values = [0.3, 0.4, 0.5]
            original = combine(*values, arm=arm)["reward"]
            values[index] = 1
            self.assertEqual(original, combine(*values, arm=arm)["reward"])

    def test_guardrails_shared_across_arms(self):
        for arm in ("bcr", "no_beat", "no_coverage", "no_richness", "beat", "coverage", "richness"):
            raw = combine(0.4, 0.4, 0.4, arm=arm)["reward"]
            guarded = combine(0.4, 0.4, 0.4, arm=arm, quality_penalty=0.1, loudness_penalty=0.2)["reward"]
            self.assertAlmostEqual(raw - guarded, 0.3)

    def test_abstention_does_not_invent_beat_score(self):
        result = combine(0.9, 0.4, 0.4, beat_scorable=False)
        self.assertEqual(result["beat_v5"], 0)
        self.assertEqual(result["beat_v5_raw"], 0.9)

    def test_coverage_cap_and_finite_contract(self):
        self.assertEqual(combine(0, 0.7, 0)["reward"], combine(0, 1, 0)["reward"])
        with self.assertRaises(ValueError):
            combine(float("nan"), 0, 0)


class ManifestTests(unittest.TestCase):
    def test_song_round_robin_preserves_all_clips_and_seed(self):
        from formal_protocol import epoch_order
        rows = [{"id": f"{song}{n}", "song": song} for song in "abc" for n in range(3)]
        order = epoch_order(rows, 13, 0)
        self.assertEqual(order, epoch_order(rows, 13, 0))
        self.assertEqual(len({r["id"] for r in order}), 9)
        for start in range(0, 9, 3):
            self.assertEqual({r["song"] for r in order[start:start + 3]}, set("abc"))

    def test_manifest_rejects_wrong_split_and_duplicates(self):
        from formal_protocol import load_manifest
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "manifest.jsonl"
            row = {"id": "x", "split": "valid"}
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaises(ValueError):
                load_manifest(path, "train")
            path.write_text((json.dumps(row) + "\n") * 2)
            with self.assertRaises(ValueError):
                load_manifest(path, "valid")


if __name__ == "__main__":
    unittest.main()
