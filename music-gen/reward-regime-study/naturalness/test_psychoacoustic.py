"""Analytical anchors and device agreement; no synthetic naturalness labels."""
import unittest
import numpy as np
from mosqito.sq_metrics import roughness_dw
from psychoacoustic_features import AuditoryGPU


class PsychoacousticContract(unittest.TestCase):
    def test_dw_asper_anchor(self):
        t=np.arange(48000)/48000
        x=(1+np.cos(2*np.pi*70*t))*np.sin(2*np.pi*1000*t)
        x*=.02/np.sqrt(np.mean(x*x)) #60dB SPL
        value=roughness_dw(x,48000)[0]
        self.assertTrue(.85<float(np.mean(value))<1.15)

    def test_auditory_modulation_and_gain_invariance(self):
        t=np.arange(48000)/48000;x=.05*np.sin(2*np.pi*1000*t)
        extractor=AuditoryGPU('cpu')
        plain=extractor(x);mod=extractor(x*(1+.8*np.cos(2*np.pi*70*t)));quiet=extractor(x*.1)
        self.assertGreater(mod['modulation_depth_20_300'],plain['modulation_depth_20_300']+.05)
        self.assertLess(abs(plain['modulation_depth_20_300']-quiet['modulation_depth_20_300']),1e-4)
        self.assertFalse(extractor(np.zeros(48000))['valid'])


if __name__=='__main__':unittest.main()
