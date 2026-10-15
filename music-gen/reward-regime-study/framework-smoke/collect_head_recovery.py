"""One-shot immutable head-ablation recovery; partial means are never final scores."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--host', required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    report = {'snapshot_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'host': a.host, 'runs': []}
    for model in ['diffrhythm2', 'heartmula', 'songbloom']:
        for reward in ['richness', 'instrument', 'arrangement']:
            folder = a.root / 'outputs' / f'{model}-{reward}-3e-4'
            if not folder.exists():
                continue
            experiment=json.loads((folder/'experiment.json').read_text())
            assert experiment['data_sha256']=='c39d43e81793bee3f312c7028a0483b5da41068db4544acd258115043101e46f'
            journal = folder / 'steps.jsonl'
            rows = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
            for i, row in enumerate(rows):
                assert row['step'] == i + 1
                assert sha(folder / 'checkpoints' / f'step_{i+1:06d}.pt') == row['checkpoint_sha256']
                for j, score in enumerate(row['components']):
                    audio = folder / 'train_rollouts' / f'{i+1:06d}' / str(j) / 'audio.flac'
                    assert sha(audio) == score['audio_sha256'], str(audio)
            target = a.output / f'{model}-{reward}'
            target.mkdir(exist_ok=True)
            for name in ['steps.jsonl', 'experiment.json', 'completion.json']:
                if (folder / name).exists():
                    shutil.copyfile(folder / name, target / name)
            validation = []
            for valid in sorted(folder.glob('valid_*')):
                receipt = valid / 'receipt.json'
                if not receipt.exists():
                    continue
                value = json.loads(receipt.read_text())
                assert value['n'] == 30 and len(value['rows']) == 30
                for i, row in enumerate(value['rows']):
                    assert sha(Path(row.get('private_audio', valid / f'{i:03d}' / 'audio.flac'))) == row['audio_sha256']
                shutil.copyfile(receipt, target / f'{valid.name}.json')
                validation.append({'step': value['step'], 'n': 30, 'mean': value['mean_reward'],
                                   'receipt_sha256': sha(receipt)})
            report['runs'].append({'model': model, 'reward': reward, 'steps': len(rows), 'complete': False,
                'scope': 'Head-only ablation; superseded for backbone conclusions; preserved, not resumed',
                'validation': validation,
                'trace': [{'step': r['step'], 'reward': r['reward'], 'kl': r['sampled_kl']} for r in rows],
                'kl_unit': experiment['policy']['kl_unit'],
                'checkpoint_audio_binding': 'all journaled checkpoints and sampled PCM24-FLAC SHA256 verified'})
    (a.output / 'progress.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps([{k: r[k] for k in ['model', 'reward', 'steps', 'validation']} for r in report['runs']]))


if __name__ == '__main__':
    main()
