"""Boundary tests protect reward sign, rejection and calibration contracts."""
import unittest
import numpy as np
from reward_adapter import CalibratedReward, aggregate_windows, scale
from signal_features import features


class Contract(unittest.TestCase):
    def test_silence_has_no_fabricated_spectral_naturalness(self):
        r=features(np.zeros((48000,2)),48000)
        self.assertFalse(r['valid']); self.assertIsNone(r['flatness'])

    def test_scale_and_invalid_calibration(self):
        self.assertEqual(scale(1,0,1),1)
        self.assertEqual(scale(-1,0,1),-1)
        for x,lo,hi in [(1,1,1),(float('nan'),0,1)]:
            with self.assertRaises(ValueError): scale(x,lo,hi)

    def test_antiphase_stereo_retains_valid_energy_without_nan_spectrum(self):
        x=.1*np.sin(2*np.pi*220*np.arange(48000)/48000)
        r=features(np.column_stack([x,-x]),48000)
        self.assertTrue(r['valid']);self.assertIsNone(r['flatness'])
        self.assertEqual(r['mono_fold_down_energy_ratio'],0)

    def test_rejection_and_low_quality_cannot_create_positive_bonus(self):
        r=CalibratedReward(0,10,0,10,0,10)
        args=dict(existing=5,naturalness_logit=10,pq=0,valid=True)
        self.assertLess(r.compose(**args)['reward'],0)
        self.assertEqual(r.compose(**(args|dict(valid=False)))['reward'],-1)
        self.assertEqual(r.compose(**(args|dict(duration_ok=False)))['reward'],-1)
        self.assertEqual(r.compose(**(args|dict(activity_ok=False)))['reward'],-1)

    def test_poor_window_is_retained(self):
        self.assertLess(aggregate_windows([1,1,-1]),np.mean([1,1,-1]))
        with self.assertRaises(ValueError): aggregate_windows([1,float('nan')])


if __name__=='__main__': unittest.main()
