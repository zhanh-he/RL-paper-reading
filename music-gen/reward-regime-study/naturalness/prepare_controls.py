"""Make explicit paired perturbations; do not assign perceptual ground truth.

Source manifest paths are private local files. All scoring WAVs remain outside
the repository. Constructed synths are intended style controls, not human labels.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from signal_features import sha256


def matched_rms(y, x):
    old = np.sqrt(np.mean(x*x)); new = np.sqrt(np.mean(y*y))
    return y * old/max(new, 1e-12)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sources', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--manifest', required=True)
    args = p.parse_args()
    dest = Path(args.output_dir).resolve(); dest.mkdir(parents=True, exist_ok=True)
    sources = json.loads(Path(args.sources).read_text())['samples']
    rows = []
    for row in sources:
        x, sr = sf.read(row['path'], always_2d=True)
        offset = float(row.get('offset_s', 0)); x = x[int(offset*sr):int((offset+10)*sr)]
        # Common stereo 48kHz float WAV with no gain adjustment; models resample.
        from math import gcd
        div = gcd(sr, 48000); x = signal.resample_poly(x, 48000//div, sr//div, axis=0)
        if x.shape[1] == 1: x = np.repeat(x, 2, axis=1)
        path = dest/(row['id']+'.wav'); sf.write(path, x, 48000, subtype='FLOAT')
        rows.append(row | dict(path=str(path), source_path=row['path'],
                              source_original_sha256=row.get('source_original_sha256', sha256(row['path'])),
                              offset_s=offset, excerpt_sha256=sha256(path), transform='none',
                              duration_s=len(x)/48000, sample_rate=48000))
    # Probe one generated and one recorded reference, to expose content interaction.
    bases = [next(r for r in rows if r['kind']=='generated')]
    refs = [r for r in rows if r['kind']=='recorded_reference']
    if refs: bases.append(refs[0])
    for base in bases:
        x, sr = sf.read(base['path'], always_2d=True)
        t = np.arange(len(x))/sr
        transforms = {
            'gain_minus20db': (x*.1, 'Amplitude ×0.1; no content change'),
            'lowpass_3khz': (signal.sosfiltfilt(signal.butter(6, 3000, fs=sr, output='sos'), x, axis=0), '6th order Butterworth zero-phase 3 kHz lowpass'),
            'compressed': (matched_rms(np.sign(x)*np.minimum(abs(x), .08)*(np.maximum(abs(x), .08)/.08)**.25, x), 'Static threshold 0.08, 4:1 amplitude compression; RMS matched'),
            'distorted': (matched_rms(np.clip(x*8, -.15, .15), x), 'Hard clip gain8 threshold0.15; RMS matched'),
            'metallic_mod40hz': (x*(.55+.45*np.cos(2*np.pi*40*t[:,None])), '40 Hz amplitude modulation, envelope0.10..1.0; no added gain, compare equal-RMS mod40hz_gain_control; not a naturalness label'),
        }
        mod=transforms['metallic_mod40hz'][0]
        gain=float(np.sqrt(np.mean(mod*mod))/max(np.sqrt(np.mean(x*x)),1e-12))
        transforms['mod40hz_gain_control']=(x*gain, f'Pure gain {gain:.12g}; RMS matches mod40Hz stimulus, isolates modulation from gain')
        for name, (y, description) in transforms.items():
            rid = base['id']+'__'+name; path=dest/(rid+'.wav'); sf.write(path, y, sr, subtype='FLOAT')
            rows.append(base | dict(id=rid, label=base['label']+' / '+name, path=str(path),
                                    kind='perturbation', base_id=base['id'], transform=name,
                                    matched_comparator_id=base['id']+'__mod40hz_gain_control' if name=='metallic_mod40hz' else base['id'],
                                    transform_description=description, excerpt_sha256=sha256(path),
                                    selection_reason='Paired reward-shortcut probe; no human ground truth'))
    sr=48000; t=np.arange(10*sr)/sr
    for name in ['silence', 'normal_synth', 'bright_synth']:
        x=np.zeros_like(t)
        if name!='silence':
            for k, f in enumerate([220, 277.1826, 329.6276, 440, 369.9944]):
                idx=(t>=k*2)&(t<(k+1)*2); phase=t[idx]-k*2
                envelope=np.minimum(phase/.04, 1)*np.minimum((2-phase)/.15, 1)
                harmonics=range(1, 19 if name=='bright_synth' else 5)
                wave=sum(np.sin(2*np.pi*h*f*phase)/h**(1.15 if name=='bright_synth' else 2) for h in harmonics)
                x[idx]=.12*envelope*wave
        rid='control_'+name; path=dest/(rid+'.wav'); sf.write(path, np.column_stack([x,x]),sr,subtype='FLOAT')
        rows.append(dict(id=rid,label={'silence':'静音','normal_synth':'正常加法合成器音符','bright_synth':'明亮谐波合成器音符'}[name],
                         path=str(path), kind='constructed_control', model='deterministic additive synthesis', checkpoint='not applicable',
                         input_id='five ascending/descending notes',pair_group='synth-brightness',source_asset=None,
                         source_original_sha256=sha256(path),excerpt_sha256=sha256(path), sample_rate=sr,duration_s=10,
                         transform=name,rights='Original constructed signal; CC0-1.0',
                         selection_reason='Intended style/silence counterexample; not a perceptual naturalness label'))
    Path(args.manifest).write_text(json.dumps({'schema_version':1,'samples':rows},ensure_ascii=False,indent=2)+'\n')
    print(f'{len(rows)} samples; scoring originals outside repository: {dest}')


if __name__=='__main__': main()
