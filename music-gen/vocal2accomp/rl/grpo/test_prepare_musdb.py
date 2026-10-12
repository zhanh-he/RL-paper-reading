"""Synthetic structural tests; fixture audio is not experimental training data."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import soundfile as sf
import yaml

from prepare_musdb import prepare, safe_extract


class PreparationTests(unittest.TestCase):
    def test_complete_song_split_gain_and_test_pcm_seal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "source"
            fixture = root / "fixtures"
            fixture.mkdir()
            sr = 44100
            time = np.arange(round(sr * 0.4)) / sr
            tone = np.repeat((0.02 * np.sin(2 * np.pi * 220 * time))[:, None], 2, axis=1)
            for stem in ("vocals", "drums", "bass", "other", "mixture"):
                sf.write(fixture / f"{stem}.wav", tone * (4 if stem == "mixture" else 1), sr, subtype="FLOAT")
            for split, count in (("train", 100), ("test", 50)):
                for index in range(count):
                    dest = data / split / f"{split}artist{index:03d} - song"
                    dest.mkdir(parents=True)
                    for source in fixture.iterdir():
                        (dest / source.name).symlink_to(source)
            names = sorted(x.name for x in (data / "train").iterdir())
            config = root / "config.yaml"
            config.write_text(yaml.safe_dump({"validation_tracks": names[:14]}))
            original_read = sf.read

            def guarded_read(path, **kwargs):
                if "/test/" in str(path):
                    raise AssertionError("test PCM was read")
                return original_read(path, **kwargs)

            with patch("prepare_musdb.sf.read", side_effect=guarded_read), redirect_stdout(io.StringIO()):
                result = prepare(data, config, root / "output", seconds=0.4)
            self.assertEqual(result["eligible"]["train"]["songs"], 86)
            self.assertEqual(result["eligible"]["valid"]["songs"], 14)
            self.assertFalse((root / "output/test.jsonl").exists())
            train = [json.loads(line) for line in (root / "output/train.jsonl").read_text().splitlines()]
            valid = [json.loads(line) for line in (root / "output/valid.jsonl").read_text().splitlines()]
            self.assertFalse({r["song"] for r in train} & {r["song"] for r in valid})
            vocal, _ = original_read(train[0]["vocal"])
            backing, _ = original_read(train[0]["backing"])
            np.testing.assert_allclose(backing, 3 * vocal, atol=5e-7)

    def test_zip_slip_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("../escape", "bad")
            with self.assertRaises(ValueError):
                safe_extract(archive, root / "unzip")
            self.assertFalse((root / "escape").exists())


if __name__ == "__main__":
    unittest.main()
