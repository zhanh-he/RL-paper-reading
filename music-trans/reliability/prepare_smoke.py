"""Create original procedural harmonic probes, not a singing benchmark."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    samples = []
    sr = 24000
    for i in range(12):
        pitches = np.array([60, 62, 64, 67, 65, 64, 62, 60, 67, 69, 67, 64]) + i % 5 - 2
        if i % 2:
            pitches = pitches[::-1].copy()
        onsets = 1.0 + np.arange(12) * .65
        ends = onsets + np.where(np.arange(12) % 3 == 0, .45, .3)
        wave = np.zeros(int(10 * sr), dtype=np.float32)
        for pitch, start, end in zip(pitches, onsets, ends):
            t = np.arange(round((end-start)*sr)) / sr
            envelope = np.minimum(1, t/.012) * np.minimum(1, (t[-1]-t)/.025)
            tone = sum(np.sin(2*np.pi*440*2**((int(pitch)-69)/12)*h*t) / h**1.4 for h in range(1, 6))
            a = round(start * sr)
            wave[a:a+len(t)] += (.2 * envelope * tone).astype(np.float32)
        name = f'probe-{i+1:02d}'
        wavfile.write(args.output / f'{name}.wav', sr, wave)
        np.savez(args.output / f'{name}.npz', waveform=wave)
        samples.append(dict(id=name, title=f'Procedural harmonic probe {i+1}', duration=10.,
                            npz=f'{name}.npz', wav=f'{name}.wav',
                            sha256=hashlib.sha256((args.output/f'{name}.npz').read_bytes()).hexdigest(),
                            reference=[dict(track=0,pitch=int(p),start=float(s),end=float(e)) for p,s,e in zip(pitches,onsets,ends)]))
    result = dict(id='procedural-harmonic-smoke-12', sampling_rate=sr,
                  scope='Original procedural harmonic signals; not singing, not independent musical works, not a real-world accuracy benchmark.',
                  conditions=['original','prefix-500ms','gain-minus6db','noise-snr30'], samples=samples)
    (args.output/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'samples':len(samples),'conditions':len(result['conditions']),'inferences':len(samples)*len(result['conditions'])}))


if __name__ == '__main__':
    main()
