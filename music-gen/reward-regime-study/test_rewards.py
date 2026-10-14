import unittest
import numpy as np
from rewards import spectral_richness, guard, objective


class RewardTests(unittest.TestCase):
    def test_noise_does_not_win_tonality_gate(self):
        noise = np.random.default_rng(42).normal(0,.1,48000)
        self.assertEqual(spectral_richness(noise,48000)['tonality_gate'], 0.)
        self.assertEqual(spectral_richness(np.zeros(48000),48000)['raw'], 0.)

    def test_fixed_single_targets_and_common_penalty(self):
        row = {'richness': {'raw': .4}, 'songbench': {'Instrumental': 8., 'Arrangement': 6.},
               'guard': {'invalid': False, 'penalty': .1}}
        self.assertAlmostEqual(objective(row,'richness'), .3)
        self.assertAlmostEqual(objective(row,'instrument'), .7)
        self.assertAlmostEqual(objective(row,'arrangement'), .5)
        row['guard']['invalid'] = True
        self.assertEqual(objective(row,'instrument'), 0.)

    def test_clipping_silence_and_nonfinite_are_recorded(self):
        self.assertTrue(guard(np.zeros(48000),48000)['invalid'])
        self.assertTrue(guard(np.array([np.nan]),48000)['invalid'])
        self.assertGreater(guard(np.ones(48000),48000)['clip_penalty'], 0.)


if __name__ == '__main__': unittest.main()
