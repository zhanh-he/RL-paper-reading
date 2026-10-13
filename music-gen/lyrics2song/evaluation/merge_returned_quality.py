"""Recover completed CMI scores into their original, still-unscored generation run."""

import argparse
import json
import math
from pathlib import Path

from generate_objective import sha, write


def merge(original, returned):
    if (returned.get('complete') is not True or original['generation_protocol'] != returned['generation_protocol']
            or original['generation_protocol']['dataset'] != 'cmi'):
        raise ValueError('Returned quality scores bind a different/incomplete run')
    before = {r['id']: r for r in original['rows']}
    after = {r['id']: r for r in returned['rows']}
    if len(before) < 2 or len(before) != len(original['rows']) or len(after) != len(returned['rows']) or set(before) != set(after):
        raise ValueError('Returned quality candidate coverage changed')
    indexes = {r['prompt_index'] for r in original['rows']}
    for i in indexes:
        candidates = [r for r in original['rows'] if r['prompt_index'] == i]
        if len(candidates) != 2 or {r['candidate_index'] for r in candidates} != {0, 1}:
            raise ValueError('CMI recovery must retain both fixed candidates')
    for ident, row in after.items():
        source = before[ident]
        for key in ['prompt_index', 'candidate_index', 'seed', 'audio_sha256', 'source_flac_sha256',
                    'adapter_sha256', 'seconds', 'target']:
            if row[key] != source[key]:
                raise ValueError('Returned quality identity changed: ' + key)
        if any(source['metrics']):
            raise ValueError('Do not overwrite already-scored production observations')
        for metric in ['songbench', 'mulan', 'allmusiccaps']:
            if row['metrics'].get(metric, {}).get('status') != 'complete':
                raise ValueError('Incomplete returned score: ' + metric)
        supplement = row['supplementary']
        values = [*row['metrics']['songbench']['songbench_raw'].values(),
                  *supplement['songeval'].values(), supplement['audiobox']['PQ'],
                  row['metrics']['mulan']['scores']['mulan'], row['metrics']['allmusiccaps']['scores']['allmusiccaps']]
        if len(row['metrics']['songbench']['songbench_raw']) != 7 or len(supplement['songeval']) != 5:
            raise ValueError('Scorer dimensions changed')
        if not all(math.isfinite(value) for value in values):
            raise ValueError('Non-finite returned quality score')
        row['audio_path'] = source['audio_path']
    return returned


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--returned', type=Path, required=True)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--end', type=int, default=4)
    a = p.parse_args()
    source = a.run / f'evaluation/chunk_{a.start:03d}_{a.end:03d}.json'
    original = json.loads(source.read_text())
    returned = json.loads(a.returned.read_text())
    result = merge(original, returned)
    for row in result['rows']:
        if sha(Path(row['audio_path'])) != row['audio_sha256']:
            raise ValueError('Original scoring MP3 changed during remote scoring')
    result['remote_recovery'] = {'returned_chunk_sha256': sha(a.returned), 'source_chunk_sha256': sha(source),
        'scope': 'same immutable candidates, scored remotely; no additional candidate draws'}
    write(source, result)
