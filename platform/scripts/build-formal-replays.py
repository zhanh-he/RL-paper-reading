"""Publish only audited formal traces and hash-bound generated audio."""

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
DEMOS = ROOT / 'platform/site/demos'
KEYS = ('Coherence', 'Musicality', 'Memorability', 'Clarity', 'Naturalness')
PROJECTION_SHA = 'c39d43e81793bee3f312c7028a0483b5da41068db4544acd258115043101e46f'
URL = 'https://github.com/zhanh-he/RL-paper-reading/releases/download/formal-cmi-pref-20261012/'
spec = importlib.util.spec_from_file_location('mel', Path(__file__).with_name('render-mel-visuals.py'))
mel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mel)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wave(audio, output):
    samples, rate = sf.read(audio, dtype='float32', always_2d=True)
    edges = np.linspace(0, len(samples), 1201, dtype=int)
    minimum = np.array([samples[a:b].min() for a, b in zip(edges[:-1], edges[1:])])
    maximum = np.array([samples[a:b].max() for a, b in zip(edges[:-1], edges[1:])])
    fig = plt.figure(figsize=(12, 1.6), dpi=100, facecolor='black')
    ax = fig.add_axes([.075, .19, .915, .77], facecolor='black')
    time = np.linspace(0, len(samples) / rate, len(minimum))
    ax.fill_between(time, minimum, maximum, color='#58bcbd', linewidth=0)
    ax.set_ylim(-1, 1); ax.set_xlim(0, len(samples) / rate)
    ax.set_yticks([-1, 0, 1]); ax.tick_params(colors='#d5dee6', labelsize=7, length=0)
    for spine in ax.spines.values(): spine.set_visible(False)
    fig.savefig(output, facecolor='black', dpi=100); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--staging', type=Path, required=True)
    args = parser.parse_args()
    projection = args.staging / 'projection.json'
    if digest(projection) != PROJECTION_SHA:
        raise ValueError('Dataset projection hash mismatch')
    dataset = read(projection)
    output = {'schema_version': 3, 'protocol': 'cmi-pref-triple-source-text-lyrics-baseline-9to1-v2',
              'recovered_at': '2026-10-12', 'train_conditions': 270, 'valid_conditions': 30,
              'sealed_test_conditions': 121, 'projection_sha256': PROJECTION_SHA,
              'examples': [{**{key: dataset['valid'][i][key] for key in ('condition_id', 'style', 'lyrics')},
                            'index': i, 'seed': 5101 + i} for i in (0, 10, 20)],
              'visual_protocol': {'waveform_bounds': [-1, 1], 'spectrum': 'mel',
                                  'frequency_hz': [0, 24000], 'decibels': [-80, 0], 'background': '#000000'},
              'arms': [], 'replicates': [], 'superseded_arms': []}
    media_dir = args.staging / 'release-assets'; media_dir.mkdir(exist_ok=True)
    visuals = DEMOS / 'visuals/formal'; visuals.mkdir(parents=True, exist_ok=True)
    archive = ROOT / 'music-gen/lyrics2song/rl/grpo/runs/2026-10-12-formal-recovery'; archive.mkdir(parents=True, exist_ok=True)
    assets = {}

    def clips(folder, prefix):
        rows = []
        for i in (0, 10, 20):
            path = folder / f'valid_{i:02d}'
            receipt = path / 'receipt.json'
            if not receipt.exists(): continue
            record = read(receipt); audio = path / 'audio.flac'
            sha = digest(audio)
            if sha != record['flac_sha256'] or record['seed'] != 5101 + i:
                raise ValueError(f'Invalid audio binding: {path}')
            data, rate = sf.read(audio, always_2d=True)
            if rate != 48000 or abs(len(data) / rate - record['seconds']) > 1 / rate:
                raise ValueError(f'Audio duration/rate mismatch: {path}')
            source_sha = sha
            cache = args.staging / 'mp3-192k-v1'
            cache.mkdir(exist_ok=True)
            encoded = cache / f'{source_sha}.mp3'
            if not encoded.exists():
                subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(audio),
                                '-map', '0:a:0', '-c:a', 'libmp3lame', '-b:a', '192k',
                                '-ar', '48000', '-map_metadata', '-1', str(encoded)], check=True)
            sha = digest(encoded)
            name = f'{prefix}_valid{i:02d}_{sha[:12]}'
            destination = media_dir / f'{name}.mp3'
            if not destination.exists(): shutil.copyfile(encoded, destination)
            elif digest(destination) != sha: raise ValueError('Release asset collision')
            playback, playback_rate = sf.read(destination, always_2d=True)
            if abs(len(playback) / playback_rate - record['seconds']) > .1:
                raise ValueError('MP3 transcode changed audio duration')
            for kind, render in [('wave', wave), ('spectrum', mel.render)]:
                image = visuals / f'{name}_{kind}.png'
                if not image.exists(): render(destination, image)
            flags = record.get('truncated', False)
            entry = {'index': i, 'seed': record['seed'], 'seconds': record['seconds'],
                     'sha256': sha, 'source_audio_sha256': source_sha, 'audio': URL + destination.name,
                     'encoding': 'MP3 192 kbps / 48 kHz; no loudness normalization',
                     'signal': record['signal'], 'signal_source': 'lossless original before MP3 encoding',
                     'wave': f'visuals/formal/{name}_wave.png', 'spectrum': f'visuals/formal/{name}_spectrum.png',
                     'wave_sha256': digest(visuals / f'{name}_wave.png'),
                     'spectrum_sha256': digest(visuals / f'{name}_spectrum.png'),
                     'truncated': any(flags.values()) if isinstance(flags, dict) else bool(flags),
                     'reaches_30_seconds': record['seconds'] >= 30,
                     'scored_audio_sha256': record.get('scored_audio_sha256'),
                     'reward': record.get('reward')}
            if record.get('backend'):
                entry['inference_backend'] = record['backend']
                entry['inference_dtype'] = record['dtype']
            assets[destination.name] = {'sha256': sha, 'bytes': destination.stat().st_size,
                                        'source_audio_sha256': source_sha}
            rows.append(entry)
        return rows

    canonical = args.staging / 'long/cmi_triple_9to1_songeval_lr3e-4_train_20261007_lab5090/step_000270/baseline'
    baseline = clips(canonical, 'yue2_baseline')
    if len(baseline) != 3: raise ValueError('Three canonical YuE2 baselines are required')
    for reward in ('songeval', 'musecritic'):
        host = 'lab5090' if reward == 'songeval' else 'gadi'
        for lr in ('2e-5', '1e-4', '3e-4'):
            suffix = '20261007_lab5090' if host == 'lab5090' else '20261006'
            run_name = f'cmi_triple_9to1_{reward}_lr{lr}_train_{suffix}'
            source = args.staging / host / run_name
            verified = read(source / 'verified_complete.json')
            if verified['status'] != 'verified-complete' or verified['verified_train_steps'] != 270:
                raise ValueError(f'Unaudited completed run: {source}')
            for s, value in verified['validation'].items():
                receipt = source / f'step_{int(s):06d}/receipt.json'
                if receipt.exists():
                    entries = read(receipt)['heldout']
                    value['dimensions'] = {k: sum(r['reward'][k] for r in entries) / len(entries) for k in KEYS}
            arm = {**verified, 'id': f'yue2-{reward}-{lr}', 'model': 'YuE2',
                   'reward': 'SongEval' if reward == 'songeval' else 'MuseCritic', 'lr': lr, 'host': host,
                   'checkpoints': [1, 25, 50, 100, 270], 'long_replays': {'0': baseline}}
            for s in arm['checkpoints']:
                rendered = clips(args.staging / 'long' / run_name / f'step_{s:06d}/grpo', f'yue2_{reward}_lr{lr}_step{s:06d}')
                if rendered: arm['long_replays'][str(s)] = rendered
            output['arms'].append(arm)
            (archive / f'{arm["id"]}.json').write_text(json.dumps(verified, indent=2) + '\n')
    for reward in ('songeval', 'musecritic'):
        for lr in ('1e-6', '3e-6', '1e-5'):
            source = args.staging / f'muse/{reward}_lr{lr}_train_v2'
            verified = read(source / 'verified_recovery.json')
            arm = {**{k: v for k, v in verified.items() if k != 'audio_bindings'},
                   'id': f'muse-{reward}-{lr}', 'model': 'Muse',
                   'reward': 'SongEval' if reward == 'songeval' else 'MuseCritic', 'lr': lr,
                   'host': 'gadi', 'validation': {}, 'long_replays': {}}
            for receipt in source.glob('validation/step_*/receipt.json'):
                independent = read(receipt.with_name('verified_validation.json'))
                if independent['receipt_sha256'] != digest(receipt):
                    raise ValueError('Changed Muse validation receipt after independent audit')
                evaluated = read(receipt); entries = evaluated['heldout']; s = str(int(receipt.parent.name.removeprefix('step_')))
                if evaluated['n'] != 30: raise ValueError('Incomplete Muse validation')
                arm['validation'][s] = {'mean_reward': evaluated['mean_reward'], 'n': 30,
                    'dimensions': {k: sum(r['reward'][k] for r in entries) / 30 for k in KEYS}}
            shared_receipt = args.staging / f'muse/{reward}_lr1e-6_train_v2/validation/step_000000/receipt.json'
            if '0' not in arm['validation'] and shared_receipt.exists():
                if read(shared_receipt.with_name('verified_validation.json'))['receipt_sha256'] != digest(shared_receipt):
                    raise ValueError('Shared baseline lacks independent verification')
                evaluated = read(shared_receipt)
                if evaluated['n'] != 30 or evaluated['adapter'] is not None:
                    raise ValueError('Shared Muse baseline is not a 30-condition frozen reference')
                arm['validation']['0'] = {'mean_reward': evaluated['mean_reward'], 'n': 30,
                    'dimensions': {k: sum(r['reward'][k] for r in evaluated['heldout']) / 30 for k in KEYS}}
            if arm['verified_train_steps'] == 270 and all(s in arm['validation'] for s in ('0', '270')):
                arm['status'] = 'verified-complete'
            arm['superseded_reason'] = '2026-10-12: user requested YuE2-aligned LR grid; old Muse inference and pending validation cancelled.'
            output['superseded_arms'].append(arm)
            (archive / f'{arm["id"]}.json').write_text(json.dumps(verified, indent=2) + '\n')
    for reward in ('songeval', 'musecritic'):
        for lr in ('2e-5', '1e-4', '3e-4'):
            receipt = args.staging / f'muse_aligned/submission_{reward}_lr{lr}.json'
            submitted = read(receipt)
            if submitted['standalone_inference'] or submitted['validation_generation']:
                raise ValueError('Muse standalone inference is disabled by user request')
            if submitted['verified_train_steps'] != 0 or submitted['train_curve']:
                raise ValueError('Submission receipt is not a verified training log')
            output['arms'].append(submitted)
            (archive / f'{submitted["id"]}-submission.json').write_text(json.dumps(submitted, indent=2) + '\n')
    for source in sorted((args.staging / 'gadi').glob('cmi_triple_9to1_songeval*')):
        if not (source / 'verified_complete.json').exists(): continue
        record = read(source / 'verified_complete.json')
        output['replicates'].append({**record, 'model': 'YuE2', 'reward': 'SongEval', 'lr': source.name.split('_lr')[1].split('_')[0], 'host': 'gadi · BF16'})
    (DEMOS / 'formal-9to1-runs.json').write_text(json.dumps(output, indent=2, ensure_ascii=False) + '\n')
    (media_dir / 'asset-bindings.json').write_text(json.dumps(assets, indent=2) + '\n')
    (archive / 'asset-bindings.json').write_text(json.dumps(assets, indent=2) + '\n')
    print(f'{len(output["arms"])} arms; {len(assets)} unique hash-bound audio assets')


if __name__ == '__main__':
    main()
