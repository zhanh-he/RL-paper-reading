"""Cross-host quality scoring of immutable, already PER-selected MP3s.

This is a numerical audit, not another candidate draw or extra test observation.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from generate_objective import sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--extras-python', required=True)
    p.add_argument('--songeval', type=Path, required=True)
    p.add_argument('--dataset', choices=['wsb', 'cmi'], default='wsb')
    a = p.parse_args()
    a.bundle = a.bundle.resolve()
    sys.path[:0] = [str(a.bundle), str(a.bundle / 'src'), str(a.bundle / 'vendor')]
    import wsb_eval as frozen
    from mot_infer import extended_eval
    import torch
    release = frozen.verify_release()
    assets = frozen.verify_assets(['songbench', 'mulan', 'allmusiccaps'])
    config = json.loads((a.bundle / 'configs/evaluation.json').read_text())
    sources = sorted(a.input.glob(f'*/{a.dataset}/evaluation/chunk_*.json'))
    if not sources:
        raise ValueError('No selected-audio chunks were staged')
    audit = []
    for source in sources:
        original = json.loads(source.read_text())
        report = a.output / source.relative_to(a.input).parent
        chunk = report / source.name
        cache = {r['id']: r for r in json.loads(chunk.read_text())['rows']} if chunk.exists() else {}
        rows = json.loads(json.dumps(original['rows']))
        for row in rows:
            audio = source.parent / 'audio' / row['id'] / 'audio.mp3'
            if sha(audio) != row['audio_sha256']:
                raise ValueError('Transferred scoring audio changed')
            row['audio_path'] = str(audio.resolve())
            row['metrics'] = {'per': row['metrics']['per']} if a.dataset == 'wsb' else {}
            row.pop('supplementary', None)
        for metric in ['songbench', 'mulan', 'allmusiccaps']:
            code = release['code_files']
            if metric == 'songbench':
                code = {**code, 'blackwell_extension': sha(Path(__file__).with_name('score_songbench_blackwell.py'))}
            jobs = []
            for row in rows:
                identity = extended_eval.identity(row, metric, config, assets, code)
                receipt = report / 'receipts' / row['id'] / (metric + '.json')
                if extended_eval.read_cached(receipt, identity) is None:
                    jobs.append({**row, 'identity': identity, 'result_path': str(receipt.resolve())})
            jobfile = report / (metric + '.jobs.json')
            write(jobfile, {'jobs': jobs, 'config': config})
            command = [str(a.bundle / '.venvs/metrics/bin/python'), str(a.bundle / 'wsb_eval.py'),
                       '_worker', '--metric', metric, '--jobs', str(jobfile.resolve())]
            if metric == 'songbench':
                command = [command[0], str(Path(__file__).with_name('score_songbench_blackwell.py')),
                           '--bundle', str(a.bundle), '--jobs', str(jobfile.resolve())]
            if jobs:
                with jobfile.with_suffix('.log').open('a') as log:
                    subprocess.run(command, env=frozen.environment(report.resolve()),
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
            for row in rows:
                identity = extended_eval.identity(row, metric, config, assets, code)
                value = extended_eval.read_cached(report / 'receipts' / row['id'] / (metric + '.json'), identity)
                if not value or value['status'] != 'complete':
                    raise ValueError(f'Incomplete {metric}: {row["id"]}')
                row['metrics'][metric] = frozen.canonical_songbench(value) if metric == 'songbench' else value
        for row in rows:
            cached = cache.get(row['id'], {})
            supplement = cached.get('supplementary', {})
            if (cached.get('audio_sha256') == row['audio_sha256'] and cached.get('adapter_sha256') == row['adapter_sha256']
                    and supplement.get('scorer', {}).get('scorer_code_sha256') == sha(Path(__file__).with_name('score_supplementary.py'))
                    and 'audiobox' in supplement and 'audiobox_state_sha256' in supplement):
                row['supplementary'] = supplement
        write(chunk, {'complete': True, 'rows': rows, 'audit_source_sha256': sha(source),
              'generation_protocol': original.get('generation_protocol'),
              'scope': 'CMI production quality evaluation' if a.dataset == 'cmi' else 'WSB cross-host numerical audit'})
        if not all('supplementary' in row for row in rows):
            subprocess.run([a.extras_python, str(Path(__file__).with_name('score_supplementary.py')),
                            '--chunk', str(chunk), '--songeval', str(a.songeval)], check=True)
        rescored = json.loads(chunk.read_text())['rows']
        for old, new in zip(original['rows'], rescored):
            if old['id'] != new['id'] or old['audio_sha256'] != new['audio_sha256']:
                raise ValueError('Cross-host row pairing changed')
            differences = {}
            for metric in ['mulan', 'allmusiccaps']:
                if metric in old['metrics']:
                    differences[metric] = new['metrics'][metric]['scores'][metric] - old['metrics'][metric]['scores'][metric]
            for dimension, value in old['metrics'].get('songbench', {}).get('songbench_raw', {}).items():
                differences['songbench_' + dimension] = new['metrics']['songbench']['songbench_raw'][dimension] - value
            for dimension, value in old.get('supplementary', {}).get('songeval', {}).items():
                differences['songeval_' + dimension] = new['supplementary']['songeval'][dimension] - value
            audit.append({'experiment': source.relative_to(a.input).parts[0], 'id': old['id'],
                          'audio_sha256': old['audio_sha256'], 'differences': differences})
        write(a.output / 'cross-host-audit.json', {'status': 'in-progress', 'host': __import__('os').uname().nodename,
            'gpu': torch.cuda.get_device_name(0), 'rows': audit,
            'scope': 'CMI production quality evaluation' if a.dataset == 'cmi' else 'audit only; do not append duplicate observations to cohort'})
    write(a.output / 'completion.json', {'status': 'complete', 'chunks': len(sources),
                                        'scored_candidates': len(audit), 'new_training': False})


if __name__ == '__main__':
    main()
