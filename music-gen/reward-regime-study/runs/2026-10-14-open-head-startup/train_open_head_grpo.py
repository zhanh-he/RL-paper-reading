"""Formal, restricted-head GRPO with all-condition validation and bounded replays."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import soundfile as sf
import torch
from open_head_backends import BACKENDS
from rewards import sha, score_remote

DATA_SHA = 'c39d43e81793bee3f312c7028a0483b5da41068db4544acd258115043101e46f'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def candidate(backend, row, value, folder):
    started = time.monotonic()
    audio, sr, records = backend.sample(row, value)
    if not records: raise ValueError('No sampled policy actions; cannot claim a GRPO rollout')
    if not np.isfinite(audio).all(): raise FloatingPointError('Nonfinite decoder output; stop and diagnose')
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/'audio.flac'; sf.write(path, audio, sr, subtype='PCM_24')
    torch.save(records, folder/'policy-records.pt')
    result = {'audio_sha256': sha(path), 'seed': value, 'seconds': len(audio)/sr,
              'sample_rate': sr, 'actions': len(records), 'generation_seconds': time.monotonic()-started,
              'condition_id': row['condition_id'], 'policy_scope': backend.metadata()}
    write(folder/'rollout.json', result)
    return path, records, result


def score(backend, args, paths):
    backend.park('cpu')
    try: return score_remote(args.scorer, paths, args.reward)
    finally: backend.park('cuda')


def validate(backend, data, args, step, baseline=False):
    folder = args.output/f'valid_{step:06d}'
    receipt = folder/'receipt.json'
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved['n'] != 30: raise ValueError('Partial validation receipt')
        for i, row in enumerate(saved['rows']):
            if sha(folder/f'{i:03d}'/'audio.flac') != row['audio_sha256']: raise ValueError('Validation audio changed')
        return
    previous = backend.head.active; backend.head.active = not baseline
    try:
        scores = []
        for i, row in enumerate(data):
            path, _, metadata = candidate(backend, row, 5101+i, folder/f'{i:03d}')
            result = score(backend, args, [path])[0]
            result.update(index=i, seed=5101+i)
            write(folder/f'{i:03d}'/'score.json', result); scores.append(result)
        write(receipt, {'step': step, 'n': len(scores), 'mean_reward': sum(s['mean'] for s in scores)/30,
                        'rows': scores, 'scope': 'All30 native60-second-budget validation; not human MOS or WSB test'})
    finally: backend.head.active = previous


def replay(backend, args):
    examples = json.loads(args.examples.read_text())
    inputs = {r['index']: r for r in examples['inputs']}
    assets = []
    for step in [0, 270]:
        backend.head.active = step != 0
        for index in examples['display_indices']:
            row = inputs[index]
            if args.model == 'songbloom':
                references = json.loads(args.reference_manifest.read_text())
                row = {**row, 'reference_file': references[row['condition_id']]['path']}
            path, _, metadata = candidate(backend, row, row['seed'], args.output/'replays'/f'{step:06d}'/f'{index:03d}')
            result = score(backend, args, [path])[0]
            assets.append({'index': index, 'step': step, **metadata, 'score': result,
                           'private_audio': str(path), 'kind': row['kind']})
    backend.head.active = True
    write(args.output/'replays'/'receipt.json', {'examples_per_checkpoint': 5, 'rows': assets,
          'scope': 'Three original CMI examples plus two modified count probes; never formal validation means'})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=BACKENDS, required=True); p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--sources', type=Path, required=True); p.add_argument('--data', type=Path, required=True)
    p.add_argument('--examples', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reference-manifest', type=Path)
    p.add_argument('--scorer', default='http://127.0.0.1:43140')
    p.add_argument('--reward', choices=['richness', 'instrument', 'arrangement'], required=True)
    p.add_argument('--lr', type=float, default=3e-4); p.add_argument('--steps', type=int, default=270)
    p.add_argument('--seconds', type=int, default=60); p.add_argument('--beta', type=float, default=.01)
    a = p.parse_args()
    if sha(a.data) != DATA_SHA: raise ValueError('Frozen train/valid file changed')
    data = json.loads(a.data.read_text()); assert len(data['train']) == 270 and len(data['valid']) == 30
    if a.steps != 270 or a.lr != 3e-4: raise ValueError('This submission has one fixed formal LR and270 conditions')
    if not (a.assets/a.model/'asset-receipt.json').exists(): raise ValueError('Missing verified asset receipt')
    if a.model == 'songbloom':
        if not a.reference_manifest: raise ValueError('SongBloom requires complete native reference inputs')
        references = json.loads(a.reference_manifest.read_text())
        for split in ['train', 'valid']:
            data[split] = [{**r, 'reference_file': references[r['condition_id']]['path']} for r in data[split]]
            for row in data[split]:
                info = references[row['condition_id']]
                if sha(Path(info['path'])) != info['sha256']: raise ValueError('Reference bytes changed')
    a.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    backend = BACKENDS[a.model](a.assets/a.model, a.sources, a.seconds)
    parameters = [backend.head.a, backend.head.b]
    optimizer = torch.optim.AdamW(parameters, lr=a.lr)
    manifest = {'model': a.model, 'reward': a.reward, 'lr': a.lr, 'beta': a.beta, 'seed': 42,
                'group_size': 2, 'steps': 270, 'train_n': 270, 'valid_n': 30, 'seconds_budget': a.seconds,
                'data_sha256': DATA_SHA, 'assets_receipt_sha256': sha(a.assets/a.model/'asset-receipt.json'),
                'source_sha256': {name: sha(Path(__file__).with_name(name)) for name in
                                  ['train_open_head_grpo.py', 'open_head_policy.py', 'open_head_backends.py']},
                'policy': backend.metadata(), 'trainable_parameters': sum(p.numel() for p in parameters),
                'scope': 'Restricted native output-head GRPO, not full backbone fine-tuning',
                'framework': 'PyTorch; DiffRhythm2 calls pinned official Flow-GRPO SDE; no TRL claimed',
                'clipping': 'One update/group: ratio clipping has no trust-region action at first gradient',
                'test_run': False, 'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    if (a.output/'experiment.json').exists():
        saved = json.loads((a.output/'experiment.json').read_text())
        for key in ['model', 'reward', 'lr', 'data_sha256', 'assets_receipt_sha256', 'source_sha256']:
            if saved[key] != manifest[key]: raise ValueError(f'Resume identity mismatch: {key}')
    else: write(a.output/'experiment.json', manifest)
    journal = a.output/'steps.jsonl'; start = 0
    if journal.exists():
        rows = [json.loads(line) for line in journal.read_text().splitlines()]
        if [r['step'] for r in rows] != list(range(1, len(rows)+1)): raise ValueError('Non-contiguous training log')
        start = len(rows)
        state = torch.load(a.output/'checkpoints'/f'step_{start:06d}.pt', weights_only=True)
        backend.head.restore(state['adapter']); optimizer.load_state_dict(state['optimizer'])
    for step in range(start+1, 271):
        began = time.monotonic(); row = data['train'][step-1]
        paths, records, metadata = [], [], []
        for i in range(2):
            path, record, meta = candidate(backend, row, 420000+step*2+i,
                a.output/'train_rollouts'/f'{step:06d}'/str(i))
            paths.append(path); records.append(record); metadata.append(meta)
        scores = score(backend, a, paths)
        rewards = np.array([s['mean'] for s in scores]); std = float(rewards.std())
        advantage = (rewards-rewards.mean())/(std+1e-8)
        optimizer.zero_grad(set_to_none=True); kl_sum = 0.; max_identity_error = 0.
        for i, trajectory in enumerate(records):
            for record in trajectory:
                logprob, kl = backend.terms(record)
                old = record['old_logprob'].to(logprob.device)
                error = float((logprob.detach()-old).abs()); max_identity_error = max(max_identity_error, error)
                if error > .02: raise ValueError(f'Rollout/recompute mismatch {error}; stop instead of training wrong likelihood')
                if not torch.isfinite(kl): raise FloatingPointError('Nonfinite reference KL')
                ratio = (logprob-old).exp()
                loss = (-float(advantage[i])*ratio+a.beta*kl)/(2*len(trajectory))
                loss.backward(); kl_sum += float(kl.detach())/(2*len(trajectory))
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        if step == 1 and kl_sum > 1e-6: raise ValueError('Frozen-reference identity KL failed')
        optimizer.step()
        record = {'step': step, 'condition_id': row['condition_id'], 'reward': float(rewards.mean()),
                  'rewards': rewards.tolist(), 'sampled_kl': kl_sum, 'kl_unit': backend.metadata()['kl_unit'],
                  'group_std': std, 'gradient_norm': float(norm), 'max_old_current_logprob_error': max_identity_error,
                  'components': scores, 'actions': [len(r) for r in records], 'seconds': time.monotonic()-began}
        cp = a.output/'checkpoints'/f'step_{step:06d}.pt'; cp.parent.mkdir(exist_ok=True)
        torch.save({'step': step, 'adapter': backend.head.adapter_state(), 'optimizer': optimizer.state_dict()}, cp)
        record['checkpoint_sha256'] = sha(cp)
        with journal.open('a') as stream: stream.write(json.dumps(record, allow_nan=False)+'\n')
        print(json.dumps({k: record[k] for k in ['step', 'reward', 'sampled_kl', 'gradient_norm', 'seconds']}), flush=True)
        if step in [50, 100, 270]:
            validate(backend, data['valid'], a, 0, baseline=True)
            validate(backend, data['valid'], a, step)
    replay(backend, a)
    write(a.output/'completion.json', {'training_steps': 270, 'full_valid_conditions': 30,
          'replay_examples': 5, 'test_run': False, 'policy_scope': backend.metadata()})


if __name__ == '__main__': main()
