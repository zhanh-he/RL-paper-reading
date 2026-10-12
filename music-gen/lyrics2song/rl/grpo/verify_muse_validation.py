"""Independently verify the exact scored/archive files in a Muse validation receipt."""

import argparse
import hashlib
import json
import math
from pathlib import Path

KEYS = ('Coherence', 'Musicality', 'Memorability', 'Clarity', 'Naturalness')
VALID_SHA = 'ed5c5f36af48ef6ce81d15c7b15b429fb250ec4a52a04a43e55d96ec3cafb59d'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1048576), b''):
            value.update(chunk)
    return value.hexdigest()


def verify(receipt):
    root = receipt.parent.resolve()
    data = json.loads(receipt.read_text())
    if (data['status'] != 'verified-fixed-validation' or data['n'] != 30
            or data['validation_jsonl_sha256'] != VALID_SHA
            or data['reference_audio_used'] or data['actual_input_modalities'] != ['text', 'lyrics']
            or data['max_tokens'] != 512 or not data['force_length']):
        raise ValueError('Wrong fixed validation protocol')
    if len(data['heldout']) != 30:
        raise ValueError('Incomplete fixed validation')
    bindings = []
    for index, row in enumerate(data['heldout']):
        if row['index'] != index or row['seed'] != 5101 + index:
            raise ValueError('Validation condition or seed mismatch')
        for key in ('audio', 'scored_audio'):
            file = (root / row[key]).resolve()
            if not file.is_relative_to(root) or digest(file) != row[key + '_sha256']:
                raise ValueError(f'Invalid {key} binding at {index}')
        if data['reward_backend'] == 'musecritic' and row['audio_sha256'] != row['scored_audio_sha256']:
            raise ValueError('MuseCritic archive/score format mismatch')
        reward = row['reward']
        if not all(math.isfinite(reward[k]) for k in (*KEYS, 'mean')):
            raise ValueError('Nonfinite score')
        if not math.isclose(reward['mean'], sum(reward[k] for k in KEYS) / 5, abs_tol=1e-7):
            raise ValueError('Incorrect reward mean')
        bindings.append({'index': index, 'seed': row['seed'], 'audio_sha256': row['audio_sha256'],
                         'scored_audio_sha256': row['scored_audio_sha256']})
    mean = sum(r['reward']['mean'] for r in data['heldout']) / 30
    if not math.isclose(mean, data['mean_reward'], abs_tol=1e-7):
        raise ValueError('Incorrect validation average')
    return {'status': 'independently-verified-fixed-validation', 'n': 30,
            'receipt_sha256': digest(receipt), 'mean_reward': mean, 'bindings': bindings}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path)
    args = parser.parse_args()
    result = verify(args.receipt)
    args.receipt.with_name('verified_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'bindings'}))
