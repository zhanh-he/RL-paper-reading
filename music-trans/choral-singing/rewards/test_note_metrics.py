import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("note_metrics.py")
SPEC = importlib.util.spec_from_file_location("note_metrics", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
import sys
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class NoteMetricsTest(unittest.TestCase):
    def setUp(self):
        self.reference = [
            MODULE.Note(72, 0.0, 1.0, "S"),
            MODULE.Note(64, 0.0, 1.0, "A"),
            MODULE.Note(55, 0.0, 1.0, "T"),
            MODULE.Note(43, 0.0, 1.0, "B"),
        ]

    def test_oracle_and_duplicate_penalty(self):
        oracle = MODULE.score(self.reference, self.reference)
        duplicated = MODULE.score(self.reference, self.reference + self.reference)
        self.assertEqual(oracle["track_note"]["f1"], 1)
        self.assertLess(duplicated["track_note"]["precision"], 1)

    def test_track_and_presence_penalize_voice_swap(self):
        swapped = MODULE.controlled_variants(self.reference)["swap_soprano_alto"]
        result = MODULE.score(self.reference, swapped)
        self.assertEqual(result["onset"]["f1"], 1)
        self.assertLess(result["track_note"]["f1"], 1)

    def test_coverage_alone_can_be_gamed(self):
        overfill = MODULE.controlled_variants(self.reference)["overfill_octave"]
        result = MODULE.score(self.reference, overfill)
        self.assertEqual(result["track_note"]["recall"], 1)
        self.assertLess(result["track_note"]["precision"], 1)

    def test_fragmentation_detected(self):
        fragments = MODULE.controlled_variants(self.reference)["fragment_sustained"]
        self.assertGreater(MODULE.fragmentation_rate(self.reference, fragments), 0)


if __name__ == "__main__":
    unittest.main()
