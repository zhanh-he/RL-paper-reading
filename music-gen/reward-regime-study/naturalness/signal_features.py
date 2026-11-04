"""Descriptive waveform features. None is a validated artificialness score.

Input: finite float waveform (samples[, channels]) in [-1, 1], sample rate.
No loudness normalization. Silence yields null spectral descriptors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.ndimage import median_filter


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def features(audio, sr):
    a = np.asarray(audio, dtype=np.float64)
    if a.ndim not in (1, 2) or sr <= 0 or not len(a) or not np.isfinite(a).all():
        raise ValueError("Expected nonempty finite samples[, channels] and positive sr")
    x = a.mean(axis=1) if a.ndim == 2 else a
    rms = float(np.sqrt(np.mean(a*a)))
    peak = float(np.max(np.abs(a)))
    n = max(1, int(sr * .04))
    blocks = x[:len(x)//n*n].reshape(-1, n)
    frame_rms = np.sqrt(np.mean(blocks*blocks, axis=1)) if len(blocks) else np.array([rms])
    active = frame_rms > 10**(-50/20)
    out = dict(duration_s=len(a)/sr, sample_rate=sr, channels=1 if a.ndim == 1 else a.shape[1],
               rms_dbfs=float(20*np.log10(max(rms, 1e-12))), peak=peak,
               near_full_scale_fraction=float(np.mean(np.abs(a) >= .999)),
               out_of_range_fraction=float(np.mean(np.abs(a)>1)),
               mono_fold_down_energy_ratio=float(np.mean(x*x)/max(np.mean(a*a),1e-24)) if rms>=1e-12 else None,
               active_fraction=float(active.mean()),
               crest_db=float(20*np.log10(max(peak, 1e-12)/max(rms, 1e-12))),
               valid=bool(len(a)/sr >= 1 and rms >= 1e-5))
    names = ['flatness', 'high_band_share', 'rolloff95_hz', 'spectral_flux',
             'envelope_modulation_20_100_share', 'envelope_periodicity', 'dynamic_range_db',
             'band_3_8khz_share','max_persistent_peak_fraction_3_8khz']
    mono_rms=float(np.sqrt(np.mean(x*x)))
    if rms < 1e-5 or mono_rms < 1e-8 or len(x) < 2048:
        return out | {key: None for key in names}
    fft = min(2048, 2**int(np.floor(np.log2(len(x)))))
    f, _, z = signal.stft(x, sr, nperseg=fft, noverlap=fft//2, boundary=None)
    power = np.abs(z)**2
    energy = power.sum(axis=0)
    keep = energy > max(float(energy.max())*1e-6, 1e-12)
    p = power[:, keep]
    if not p.shape[1]:
        return out | {key: None for key in names}
    eps = 1e-18
    band = p[(f >= 90) & (f <= min(16000, sr/2))]
    flatness = np.exp(np.log(band+eps).mean(axis=0)) / np.maximum(band.mean(axis=0), eps)
    shares = p / np.maximum(p.sum(axis=0), eps)
    region=(f>=3000)&(f<min(8000,sr/2))
    band_share=float(np.mean(shares[region].sum(axis=0)))
    # Fixed-frequency prominence, NOT a machine-origin/naturalness detector.
    # >=10dB over a 13-bin local median in frames with measurable band energy.
    floor=median_filter(p,size=(13,1),mode='nearest')
    peaks=p>10*np.maximum(floor,eps)
    measurable=p[region].sum(axis=0)>np.maximum(p.sum(axis=0)*1e-5,eps)
    persistence=float(peaks[region][:,measurable].mean(axis=1).max()) if region.any() and measurable.any() else None
    roll = f[np.argmax(np.cumsum(shares, axis=0) >= .95, axis=0)]
    norm = np.abs(z) / np.maximum(np.abs(z).sum(axis=0), eps)
    flux = np.sqrt(np.sum(np.diff(norm, axis=1)**2, axis=0))
    # Hilbert envelope at 1 kHz. Music rhythm/tremolo can also elevate these.
    env = np.abs(signal.hilbert(x))
    env = signal.resample_poly(env, 1000, int(sr))
    env = env - np.mean(env)
    ef, ep = signal.welch(env, 1000, nperseg=min(2048, len(env)))
    mod = float(ep[(ef >= 20)&(ef <= 100)].sum()/max(ep[(ef >= 1)&(ef <= 100)].sum(), eps))
    ac = signal.fftconvolve(env, env[::-1], mode='full')[len(env)-1:]
    periodicity = float(np.max(ac[10:min(101, len(ac))])/max(ac[0], eps))
    db = 20*np.log10(np.maximum(frame_rms[active], 1e-12))
    out.update(flatness=float(np.median(flatness)),
               band_3_8khz_share=band_share,max_persistent_peak_fraction_3_8khz=persistence,
               high_band_share=float(np.mean(shares[f >= 8000].sum(axis=0))),
               rolloff95_hz=float(np.median(roll)), spectral_flux=float(np.mean(flux)),
               envelope_modulation_20_100_share=mod, envelope_periodicity=periodicity,
               dynamic_range_db=float(np.percentile(db, 95)-np.percentile(db, 10)) if len(db) else None)
    return out


def main():
    import soundfile as sf
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    rows = []
    for row in json.loads(Path(args.manifest).read_text())['samples']:
        started = time.perf_counter()
        x, sr = sf.read(row['path'], always_2d=True)
        result = features(x, sr)
        rows.append(dict(id=row['id'], audio_sha256=sha256(row['path']), signal=result,
                         elapsed_s=time.perf_counter()-started))
    Path(args.output).write_text(json.dumps({'version':'signal-diagnostics-v1', 'rows':rows}, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
