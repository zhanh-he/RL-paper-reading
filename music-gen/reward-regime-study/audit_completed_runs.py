"""Read-only CPU audit; export scores/receipts, never prompts or lossless audio."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

DIMENSIONS = ['Vocal', 'Instrumental', 'Melody', 'Structure', 'Arrangement', 'Mixing', 'Musicality']
DATA_SHA = 'c39d43e81793bee3f312c7028a0483b5da41068db4544acd258115043101e46f'
HEAD_SHA = 'ca27966a6d2b3312587b9e1d53ee3adf6bde8ac68a7d339e6f962206cbfde312'


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()


def load(path):
    return json.loads(path.read_text())


def validate_scores(samples, scores, reward, stats):
    assert len(samples) == len(scores)
    for sample, score in zip(samples, scores):
        path = Path(sample['path'])
        digest = sha(path)
        assert digest == sample['sha256'] == score['audio_sha256'], 'Audio/score binding failed'
        assert score['reward'] == reward
        assert score['scorer_identity']['songbench_weights_sha256'] == HEAD_SHA
        assert score['scorer_identity']['scored_format'] == 'PCM24-FLAC'
        values = [score['songbench'][key] for key in DIMENSIONS]
        assert all(math.isfinite(v) for v in values)
        raw = score['richness']['raw'] if reward == 'richness' else score['songbench'][
            'Instrumental' if reward == 'instrument' else 'Arrangement']/10
        expected = 0. if score['guard']['invalid'] else raw-score['guard']['penalty']
        assert abs(expected-score['mean']) < 1e-6, 'Logged objective differs from formula'
        stats['bound_audio_n'] += 1
        stats['bound_audio_bytes'] += path.stat().st_size


def audit(folder, model):
    result = dict(model=model, run=folder.name, status='rejected', errors=[])
    try:
        experiment = load(folder/'experiment.json')
        assert experiment['data_sha256'] == DATA_SHA
        assert (experiment['train_n'], experiment['valid_n'], experiment['steps']) == (270, 30, 270)
        result['experiment'] = {key: experiment[key] for key in [
            'reward', 'lr', 'beta', 'seed', 'train_n', 'valid_n', 'steps', 'data_sha256']}
        result['experiment_sha256'] = sha(folder/'experiment.json')
        rows = [json.loads(line) for line in (folder/'steps.jsonl').read_text().split('\n') if line.strip()]
        assert [row['step'] for row in rows] == list(range(1, 271)), 'Missing/duplicate updates'
        stats = dict(bound_audio_n=0, bound_audio_bytes=0)
        trace = []
        for row in rows:
            if model == 'diffrhythm2':
                samples, scores = row['samples'], row['scores']
                assert row['policy_version'] == row['step']-1
                assert row['lr'] == .0003 and row['beta'] == .05
                assert row['metrics']['nonzero_attention'] > 0 and row['metrics']['nonzero_mlp'] > 0
                kl = row['kl']
            else:
                scores = row['components']
                samples = [dict(path=str(folder/'train_rollouts'/f'{row["step"]:06d}'/str(i)/'audio.flac'),
                                sha256=score['audio_sha256']) for i, score in enumerate(scores)]
                kl = row['sampled_kl']
            assert len(samples) == 2
            assert math.isfinite(kl) and math.isfinite(row['reward'])
            validate_scores(samples, scores, experiment['reward'], stats)
            trace.append(dict(step=row['step'], reward=row['reward'], kl=kl))
        assert abs(trace[0]['kl']) < 1e-8, 'Reference identity check failed'
        final = folder/'checkpoints'/('iter_0000270' if model == 'diffrhythm2' else 'step_000270')
        meta = load(final/'meta.json')
        assert meta.get('global_step', meta.get('step')) == 270
        if model == 'diffrhythm2':
            for part in ['model', 'optimizer', 'lr_scheduler']:
                assert list((final/part).glob('*.distcp')), f'Missing {part} checkpoint'
            assert (final/'rng.pt').is_file()
        else:
            assert (final/'adapter/adapter_model.safetensors').is_file()
            assert (final/'state.pt').is_file()
        result['checkpoint'] = dict(step=270, meta_sha256=sha(final/'meta.json'),
            files=[dict(name=str(p.relative_to(final)), bytes=p.stat().st_size, sha256=sha(p))
                   for p in sorted(final.rglob('*')) if p.is_file()])
        validation = []
        for step in [0, 50, 100, 270]:
            path = folder/f'valid_{step:06d}'/'receipt.json'
            receipt = load(path)
            scores = receipt['rows']
            assert receipt['n'] == len(scores) == 30
            samples = receipt.get('samples') or [
                dict(path=str(path.parent/f'{i:03d}'/'audio.flac'), sha256=score['audio_sha256'])
                for i, score in enumerate(scores)]
            if model == 'diffrhythm2':
                assert [s['seed'] for s in samples] == list(range(5101, 5131))
                assert all(s['policy_version'] == step for s in samples)
            validate_scores(samples, scores, experiment['reward'], stats)
            mean = sum(score['mean'] for score in scores)/30
            assert abs(mean-receipt.get('mean', receipt.get('mean_reward'))) < 1e-6
            means = {key: sum(s['songbench'][key] for s in scores)/30 for key in DIMENSIONS}
            means['SongBench_mean7'] = sum(means[key] for key in DIMENSIONS)/7
            means['SP_Richness'] = sum(s['richness']['raw'] for s in scores)/30
            validation.append(dict(step=step, n=30, target_mean=mean, means=means,
                invalid_n=sum(s['guard']['invalid'] for s in scores),
                receipt_sha256=sha(path),
                rows=[dict(index=i, audio_sha256=s['audio_sha256'], seconds=s['seconds'],
                           songbench=s['songbench'], mean7=sum(s['songbench'][k] for k in DIMENSIONS)/7,
                           target=s['mean'], sp=s['richness']['raw'], guard=s['guard'])
                      for i, s in enumerate(scores)]))
        assert (folder/'completion.json').is_file()
        result.update(status='verified-historical-comparator', updates=270, validation=validation,
                      trace=trace, binding=stats, steps_sha256=sha(folder/'steps.jsonl'),
                      kl_unit='conditional Gaussian KL / active latent dimension' if model == 'diffrhythm2'
                      else 'sampled k3 KL / semantic audio token')
    except Exception as error:
        result['errors'].append(f'{type(error).__name__}: {error}')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diff-root', type=Path, required=True)
    parser.add_argument('--yue-root', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    args = parser.parse_args()
    assert sha(args.data) == DATA_SHA
    data = load(args.data)
    assert len(data['train']) == 270 and len(data['valid']) == 30
    runs = []
    for model, root in [('diffrhythm2', args.diff_root), ('yue2', args.yue_root)]:
        for reward in ['richness', 'instrument', 'arrangement']:
            suffix = '-beta5e2' if model == 'diffrhythm2' else ''
            runs.append(audit(root/f'{model}-{reward}-lr3e4{suffix}-seed42', model))
    baselines = {}
    for run in runs:
        if run['status'] != 'verified-historical-comparator':
            continue
        binding = [r['audio_sha256'] for r in run['validation'][0]['rows']]
        if run['model'] in baselines and binding != baselines[run['model']]:
            run['status'] = 'rejected'
            run['errors'].append('Different frozen baseline across reward arms')
        baselines.setdefault(run['model'], binding)
    print(json.dumps(dict(checked_at=datetime.now(timezone.utc).isoformat(),
        scope='CPU/filesystem audit, no new generation or scoring. Historical subhead + guard experiments, NOT seven-mean training.',
        expected_data_sha256=DATA_SHA, new_training_started=False, runs=runs), allow_nan=False))
