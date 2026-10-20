"""CPU-only checks of publication/provenance, not a model acceptance test."""
import hashlib
import json
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parent


class ContractTest(unittest.TestCase):
    def test_lr_matches_current_yue2_group(self):
        protocol = json.loads((ROOT.parent/'protocol.json').read_text())
        self.assertEqual(protocol['design']['lr_grid'], ['3e-4'])
        self.assertEqual(protocol['backbone_round']['lr'], 3e-4)
        self.assertEqual(protocol['backbone_round']['beta'], .05)
        self.assertEqual(protocol['backbone_round']['scheduler'], 'constant')
        self.assertEqual(protocol['models'][-1]['id'], 'private')

    def test_real_miles_receipt(self):
        record = json.loads((ROOT/'diffrhythm2-miles-acceptance.json').read_text())
        self.assertTrue(record['full_framework_e2e'])
        self.assertFalse(record['formal_training'])
        self.assertEqual(record['before'], record['after'])
        self.assertEqual(record['before']['global_step'], 2)
        self.assertEqual(record['first']['kl'], 0)
        self.assertGreater(record['second']['kl'], 0)
        for point in [record['first'], record['second']]:
            self.assertEqual(point['lr'], 3e-4)
            self.assertEqual(point['beta'], .05)
            self.assertGreater(point['metrics']['nonzero_attention'], 0)
            self.assertGreater(point['metrics']['nonzero_mlp'], 0)
            for audio, score in zip(point['samples'], point['scores']):
                self.assertEqual(audio['sha256'], score['audio_sha256'])

    def test_captured_source_and_queue_parameters(self):
        receipt = json.loads((ROOT/'launch-receipt.json').read_text())
        for name, expected in receipt['source_sha256'].items():
            # This immutable receipt describes the original 2026-10-16 submission.
            source = subprocess.check_output(['git', 'show',
                'c65498a:music-gen/reward-regime-study/backbone-rl/'+name], cwd=ROOT)
            self.assertEqual(hashlib.sha256(source).hexdigest(), expected, name)
        self.assertEqual(len(receipt['jobs']), 6)
        for job in receipt['jobs']:
            self.assertEqual(job['lr'], 3e-4)
            self.assertEqual(job['beta'], .05)
            self.assertEqual(job['scheduler'], 'constant')


if __name__ == '__main__':
    unittest.main()
