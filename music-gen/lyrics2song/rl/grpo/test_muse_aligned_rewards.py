"""Exercise guard ordering and strictness without models or GPUs."""

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf


class GuardTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        fake = self.root / 'base.py'
        fake.write_text('''import re
def _count_audio_tokens(text): return len(re.findall(r'<AUDIO_\\d+>', text))
class MusecriticRM:
    def _handle_failure(self, exc, request_id): raise exc
''')
        swift = types.ModuleType('swift')
        rewards = types.ModuleType('swift.rewards')
        rewards.orms = {}
        self.modules = patch.dict(sys.modules, {'swift': swift, 'swift.rewards': rewards})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.env = patch.dict(os.environ, {'MUSE_BASE_PLUGIN': str(fake),
                            'MUSE_INVALID_OUTPUT_LOG': str(self.root / 'invalid.jsonl')})
        self.env.start()
        self.addCleanup(self.env.stop)
        spec = importlib.util.spec_from_file_location('aligned_test', Path(__file__).with_name('muse_cmi_aligned_rewards.py'))
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def guard(self, seconds=2, nonfinite=False, failure=False):
        path = self.root / 'audio.wav'
        samples = np.zeros(int(seconds * 24000), dtype=np.float32)
        if nonfinite: samples[0] = np.nan
        sf.write(path, samples, 24000, subtype='FLOAT')
        class Decoder:
            calls = []
            def _decode_completions_to_wavs(self, items):
                self.calls.extend(items)
                if failure: raise RuntimeError('service unavailable')
                return [{'audio_path': str(path), 'error': None} for _ in items]
            def _handle_failure(self, exc, request_id): raise exc
        class Guard(self.module.InvalidOutputGuard, Decoder): pass
        return Guard()

    def test_order_and_zero_penalty(self):
        guard = self.guard()
        rows = guard._decode_completions_to_wavs([('a', ''), ('b', '<AUDIO_1>'), ('c', 'text')])
        self.assertEqual([r['error'] is None for r in rows], [False, True, False])
        self.assertEqual(guard.calls, [('b', '<AUDIO_1>')])
        self.assertEqual(guard._handle_failure(rows[0]['error'], 'a'), 0)
        self.assertEqual(len((self.root / 'invalid.jsonl').read_text().splitlines()), 2)

    def test_short(self):
        self.assertIsInstance(self.guard(seconds=.02)._decode_completions_to_wavs([('a', '<AUDIO_1>')])[0]['error'], self.module.InvalidCompletion)

    def test_nonfinite(self):
        self.assertIsInstance(self.guard(nonfinite=True)._decode_completions_to_wavs([('a', '<AUDIO_1>')])[0]['error'], self.module.InvalidCompletion)

    def test_infrastructure_stays_fatal(self):
        with self.assertRaisesRegex(RuntimeError, 'service unavailable'):
            self.guard(failure=True)._decode_completions_to_wavs([('a', '<AUDIO_1>')])
        with self.assertRaisesRegex(RuntimeError, 'scorer unavailable'):
            self.guard()._handle_failure(RuntimeError('scorer unavailable'), 'a')


if __name__ == '__main__': unittest.main()
