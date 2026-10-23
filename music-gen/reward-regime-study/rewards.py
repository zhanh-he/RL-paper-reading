"""Versioned spectral proxy and shared, explicitly logged safety penalties."""
import hashlib
import json
import math
import urllib.request
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as source:
        while block := source.read(8 << 20):
            h.update(block)
    return h.hexdigest()


def spectral_richness(audio, sr):
    x = audio.mean(axis=1) if audio.ndim == 2 else audio
    n = 4096
    if len(x) < n:
        return {'raw': 0., 'layer_activity': 0., 'layer_movement': 0., 'active_fraction': 0., 'tonality_gate': 0.}
    frames = x[:len(x)//n*n].reshape(-1, n)
    spectrum = np.abs(np.fft.rfft(frames, axis=1)) + 1e-8
    freq = np.fft.rfftfreq(n, 1/sr)
    edges = [60, 250, 700, 1800, 4500, 8000]
    bands = np.stack([np.sqrt(2*np.sum(spectrum[:, (freq>=lo)&(freq<hi)]**2, axis=1))/n
                      for lo, hi in zip(edges[:-1], edges[1:])], axis=1)
    shares = bands**2 / np.maximum(np.sum(bands**2, axis=1, keepdims=True), 1e-12)
    active = np.sqrt(np.mean(frames**2, axis=1)) > .01
    activity = float(np.mean(np.minimum(((bands>.002)&(shares>.10)).sum(axis=1), 3)/3*active))
    movement = np.abs(np.diff(np.log1p(100*bands), axis=0))
    movement = float(np.minimum(movement/.7, 1).mean()) if len(movement) else 0.
    region = spectrum[:, (freq>=90)&(freq<8000)]
    flatness = float(np.mean(np.exp(np.log(region).mean(axis=1))/region.mean(axis=1)))
    gate = float(np.clip((.40-flatness)/.25, 0, 1))
    return {'raw': gate*(.60*activity+.25*movement+.15*float(active.mean())),
            'layer_activity': activity, 'layer_movement': movement,
            'active_fraction': float(active.mean()), 'tonality_gate': gate}


def guard(audio, sr):
    if not np.isfinite(audio).all():
        return {'invalid': True, 'reason': 'nonfinite audio', 'penalty': 0.}
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if len(audio) else 0.
    invalid = len(audio)/sr < 1 or rms < 1e-5
    clipping = float(np.mean(np.abs(audio)>=.999)) if len(audio) else 0.
    db = float(20*np.log10(max(rms, 1e-9)))
    clip_penalty = .5*min(1., clipping/.01)
    loudness_penalty = min(.1, .01*(max(0., db+6)+max(0., -40-db)))
    return {'invalid': invalid, 'reason': 'under one second or silent' if invalid else '',
            'penalty': clip_penalty+loudness_penalty, 'clip_penalty': clip_penalty,
            'loudness_penalty': loudness_penalty, 'clipping_fraction': clipping, 'rms_dbfs': db,
            'peak': float(np.max(np.abs(audio))) if len(audio) else 0.}


def objective(result, reward):
    if reward == 'mean7':
        if result.get('technical_invalid'):
            return 0.
        dimensions = ['Vocal', 'Instrumental', 'Melody', 'Structure', 'Arrangement', 'Mixing', 'Musicality']
        values = [float(result['songbench'][key]) for key in dimensions]
        if not all(math.isfinite(value) for value in values):
            raise ValueError('Missing/nonfinite SongBench head')
        return sum(values)/7
    if reward not in ['richness', 'instrument', 'arrangement']:
        raise ValueError('Unknown objective')
    if result['guard']['invalid']:
        return 0.
    if reward == 'richness':
        raw = result['richness']['raw']
    else:
        raw = result['songbench']['Instrumental' if reward == 'instrument' else 'Arrangement']/10
    value = raw - result['guard']['penalty']
    if not math.isfinite(value):
        raise ValueError('Nonfinite target reward')
    return value


def score_remote(url, paths, reward):
    request = urllib.request.Request(url.rstrip('/')+'/score',
        data=json.dumps({'paths': [str(p) for p in paths], 'reward': reward}).encode(),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=1800) as response:
        result = json.load(response)
    if len(result['rows']) != len(paths):
        raise ValueError('Reward response count mismatch')
    for path, row in zip(paths, result['rows']):
        if row['audio_sha256'] != sha(path):
            raise ValueError('Reward/audio identity mismatch')
    return result['rows']
