"""One allocated GPU owns matched model arms on a disjoint condition shard."""

import argparse
import json
import os
import subprocess
from pathlib import Path

from generate_objective import write
from pack_completed_chunk import pack, verified_archive

ARMS = ['baseline', 'songeval-2e-5', 'musecritic-2e-5', 'songeval-1e-4',
        'musecritic-1e-4', 'songeval-3e-4', 'musecritic-3e-4']


def bounds(start, end, n):
    return [(start + (end-start)*i//n, start + (end-start)*(i+1)//n) for i in range(n)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--storage', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--songeval', type=Path, required=True)
    p.add_argument('--python', required=True)
    p.add_argument('--task', type=int, required=True)
    p.add_argument('--tasks', type=int, default=12)
    a = p.parse_args()
    if not 0 <= a.task < a.tasks:
        p.error('Invalid task index')
    if not (a.work / 'preparation-receipt.json').is_file():
        raise FileNotFoundError('Preparation has not completed')
    import torch
    memory = torch.cuda.get_device_properties(0).total_memory
    if memory < 79 * (1 << 30):
        raise RuntimeError('Full Q3O worker requires an 80GB-class GPU')
    code = Path(__file__).parent
    output = a.storage / f'shard_{a.task:02d}'
    bundle = a.work / 'wsb_metrics'
    write(output / 'assignment.json', {'task': a.task, 'tasks': a.tasks, 'host': os.uname().nodename,
        'job': os.environ.get('PBS_JOBID'), 'gpu': torch.cuda.get_device_name(0),
        'gpu_bytes': memory, 'torch': torch.__version__, 'wsb': bounds(4, 192, a.tasks)[a.task],
        'cmi': bounds(4, 121, a.tasks)[a.task], 'arms': ARMS, 'new_training': False,
        'reserved_lab5090_ranges': {'wsb': [0, 4], 'cmi': [0, 4]}})
    for dataset, end in [('wsb', 192), ('cmi', 121)]:
        lower, upper = bounds(4, end, a.tasks)[a.task]
        source = code / ('reproduction_manifest.jsonl' if dataset == 'wsb' else 'cmi_test.jsonl')
        split = ['--split', str(code / 'cmi-pref-triple-9to1-split.json')] if dataset == 'cmi' else []
        for start in range(lower, upper, 4):
            stop = min(start + 4, upper)
            for arm in ARMS:
                run = output / f'yue2-{arm}' / dataset
                chunk = run / 'evaluation' / f'chunk_{start:03d}_{stop:03d}.json'
                if verified_archive(run, chunk):
                    continue
                adapter = ['--adapter', str(code / 'adapters' / arm)] if arm != 'baseline' else []
                common = ['--source', str(source), *split, '--start', str(start), '--end', str(stop)]
                subprocess.run([a.python, '-u', str(code / 'generate_objective.py'), '--dataset', dataset,
                    *common, '--model', str(a.model), '--vae', str(a.work / 'models/YuE2-Vae-legacy'),
                    '--arm', f'YuE2-{arm}', *adapter, '--output', str(run)], check=True)
                subprocess.run([a.python, '-u', str(code / 'score_objective.py'), '--bundle', str(bundle),
                    '--run', str(run), *common, '--extras-python', str(a.work / 'extras/bin/python'),
                    '--songeval', str(a.songeval)], check=True)
                if dataset == 'wsb':
                    subprocess.run([a.python, '-u', str(code / 'score_q3o.py'), '--bundle', str(bundle),
                        '--chunk', str(run / 'evaluation' / f'chunk_{start:03d}_{stop:03d}.json')], check=True)
                pack(run, chunk)
    write(output / 'completion.json', {'status': 'complete', 'task': a.task, 'new_training': False})


if __name__ == '__main__':
    main()
