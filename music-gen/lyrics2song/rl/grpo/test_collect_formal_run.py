import json
import tempfile
import unittest
from pathlib import Path

from collect_formal_run import (DATASET_SHA256, TRIPLE_PROJECTION_SHA256,
                                TRIPLE_SOURCE_SHA256, collect, digest)


class CollectFormalRunTests(unittest.TestCase):
    def test_complete_run_requires_all_audio_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "audio.flac").write_bytes(b"audited audio")
            audio_hash = digest(root / "audio.flac")
            config = {
                "dataset_sha256": DATASET_SHA256,
                "dataset_protocol": "cmi-pref-lyrics-no-ref-decontaminated-v1",
                "reward_backend": "musecritic",
                "train_prompt_count": 234,
                "validation_prompt_count": 59,
                "kl_beta": 0.01,
                "compute_dtype": "bf16",
                "smoke_only": False,
                "max_tokens": 600,
                "learning_rate": 2e-5,
                "eval_seed_base": 5101,
            }
            (root / "experiment.json").write_text(json.dumps(config))
            entries = [{
                "index": index, "seed": 5101 + index,
                "audio": "audio.flac", "audio_sha256": audio_hash,
                "scored_audio": "audio.flac", "scored_audio_sha256": audio_hash,
                "reward": {"mean": 3.0},
                "signal": {"peak": 0.8, "near_full_scale_fraction": 0.0},
                "truncated": False,
            } for index in range(59)]
            for step in (0, 1, 25, 50, 100):
                target = root / f"step_{step:06d}"
                target.mkdir()
                (target / "receipt.json").write_text(json.dumps({
                    "optimizer_step": step, "heldout": entries, "mean_reward": 3.0,
                }))
            with (root / "steps.jsonl").open("w") as target:
                for step in range(1, 101):
                    target.write(json.dumps({
                        "optimizer_step": step, "rewards": [2.0, 4.0],
                        "sampled_kl": 0.01,
                        "scored_audio": [{"path": "audio.flac", "sha256": audio_hash}] * 2,
                    }) + "\n")
            result = collect(root, "musecritic", 2e-5)
            self.assertEqual(result["status"], "verified-complete")
            self.assertEqual(len(result["train_curve"]), 100)
            (root / "audio.flac").write_bytes(b"changed audio")
            with self.assertRaisesRegex(ValueError, "mismatched scored audio"):
                collect(root, "musecritic", 2e-5)

    def test_triple_source_projection_requires_all_sixty_validation_items(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "audio.flac").write_bytes(b"audited audio")
            audio_hash = digest(root / "audio.flac")
            config = {
                "dataset_sha256": TRIPLE_PROJECTION_SHA256,
                "dataset_protocol": "cmi-pref-triple-source-text-lyrics-baseline-v1",
                "source_manifest_sha256": TRIPLE_SOURCE_SHA256,
                "input_modalities": ["text", "lyrics"], "reference_audio_used": False,
                "reward_backend": "musecritic", "train_prompt_count": 240,
                "validation_prompt_count": 60, "kl_beta": 0.01,
                "compute_dtype": "bf16", "smoke_only": False, "max_tokens": 600,
                "learning_rate": 2e-5, "eval_seed_base": 5101,
            }
            (root / "experiment.json").write_text(json.dumps(config))
            entries = [{"index": index, "seed": 5101 + index,
                        "audio": "audio.flac", "audio_sha256": audio_hash,
                        "scored_audio": "audio.flac", "scored_audio_sha256": audio_hash,
                        "reward": {"mean": 3.0},
                        "signal": {"peak": 0.8, "near_full_scale_fraction": 0.0},
                        "truncated": False} for index in range(60)]
            target = root / "step_000000"
            target.mkdir()
            (target / "receipt.json").write_text(json.dumps({
                "optimizer_step": 0, "heldout": entries, "mean_reward": 3.0,
            }))
            result = collect(root, "musecritic", 2e-5)
            self.assertEqual(result["status"], "partial-verified")
            self.assertEqual(result["validation"]["0"]["n"], 60)
            config["reference_audio_used"] = True
            (root / "experiment.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "reference_audio_used"):
                collect(root, "musecritic", 2e-5)


if __name__ == "__main__":
    unittest.main()
