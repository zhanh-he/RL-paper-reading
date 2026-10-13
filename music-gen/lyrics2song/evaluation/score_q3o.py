"""Frozen paper Q3O on already selected WSB audio; never resample or train."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from generate_objective import sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--chunk', type=Path, required=True)
    a = p.parse_args()
    a.bundle = a.bundle.resolve()
    sys.path[:0] = [str(a.bundle), str(a.bundle / 'src'), str(a.bundle / 'vendor')]
    import wsb_eval as frozen
    from mot_infer import extended_eval
    release = frozen.verify_release()
    assets = frozen.verify_assets(['q3o'])
    config = json.loads((a.bundle / 'configs/evaluation.json').read_text())
    weights = {(r['prompt_index'], r['candidate_index']): r for r in
        frozen.read_jsonl(a.bundle / 'data/wsb_q3o_candidate_weights.jsonl')}
    data = json.loads(a.chunk.read_text())
    rows, jobs = [], []
    report = a.chunk.parent
    for saved in sorted(data['rows'], key=lambda r: r['prompt_index']):
        if not saved['id'].startswith('wsb_') or 'per' not in saved['metrics']:
            raise ValueError('Q3O requires the PER-selected WSB candidates')
        row = {k: v for k, v in saved.items() if k not in {'metrics', 'supplementary'}}
        bound = weights[(row['prompt_index'], row['candidate_index'])]
        if row['seed'] != bound['seed'] or sha(row['audio_path']) != row['audio_sha256']:
            raise ValueError('Q3O candidate/audio identity changed')
        for key in ['control_prompt', 'tags_full', 'genre_raw']:
            if row['target'][key] != bound['target'][key]:
                raise ValueError('Q3O target changed')
        row.update(clip_id=bound['sample_id'], control_lang=bound['target']['control_lang'])
        row['target'] = {**row['target'], 'title': bound['target']['title']}
        identity = extended_eval.identity(row, 'q3o', config, assets, release['code_files'])
        path = report / 'receipts' / row['id'] / 'q3o.json'
        rows.append((saved, row, identity, path))
        if extended_eval.read_cached(path, identity) is None:
            jobs.append({**row, 'identity': identity, 'result_path': str(path.resolve())})
    jobfile = report / f'{a.chunk.stem}.q3o.jobs.json'
    write(jobfile, {'jobs': jobs, 'config': config})
    if jobs:
        with jobfile.with_suffix('.log').open('a') as log:
            subprocess.run([str(a.bundle / '.venvs/metrics/bin/python'), str(a.bundle / 'wsb_eval.py'),
                '_worker', '--metric', 'q3o', '--jobs', str(jobfile.resolve())],
                env=frozen.environment(report.resolve()), stdout=log, stderr=subprocess.STDOUT, check=True)
    for saved, row, identity, path in rows:
        value = extended_eval.read_cached(path, identity)
        if not value or value['status'] != 'complete':
            raise ValueError(f'Incomplete Q3O: {row["id"]}')
        saved['metrics']['q3o'] = value
    data['q3o_protocol'] = {'code_files': release['code_files'], 'asset_hashes': assets,
        'config': config['q3o'], 'frozen_candidate_weights_sha256': sha(a.bundle / 'data/wsb_q3o_candidate_weights.jsonl'),
        'scope': 'sharded local-checkpoint extension; not full-sequence paper acceptance',
        'scorer_code_sha256': sha(Path(__file__))}
    write(a.chunk, data)
    print(f'{a.chunk.name}: Q3O complete on {len(rows)} selected candidates', flush=True)


if __name__ == '__main__':
    main()
