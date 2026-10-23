"""A40ms failure becomes an explicit invalid outcome, never seven fake scores."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
import soundfile as sf
from reward_server import Scorer


class Dummy:
    def cpu(self):return self
    def score(self,*args):raise AssertionError('MuQ must not receive unsupported40ms audio')


class TechnicalOutcome(unittest.TestCase):
    def test_no_stft_crash_or_fabricated_heads(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/'short.flac'
            sf.write(path,np.full((1920,2),.1,dtype=np.float32),48000,subtype='PCM_24')
            scorer=Scorer.__new__(Scorer);scorer.mean7_only=True;scorer.sep=None
            scorer.identity={'test':'technical-only'};scorer.output=root
            scorer.bench=Dummy();scorer.bench.muq=Dummy();scorer.bench.reward=Dummy()
            row=scorer.score([path],'mean7')[0]
            self.assertTrue(row['technical_invalid']);self.assertEqual(row['mean'],0.)
            self.assertTrue(all(v is None for v in row['songbench'].values()))
            self.assertAlmostEqual(row['seconds'],.04)


if __name__=='__main__':unittest.main()
