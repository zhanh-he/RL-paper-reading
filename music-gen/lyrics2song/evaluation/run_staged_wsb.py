"""Finish the 5090-owned first four conditions on an 80GB Gadi GPU."""

import argparse
import json
import subprocess
from pathlib import Path

from generate_objective import sha, write
from run_cluster_shard import ARMS, clean_install_cache
from pack_completed_chunk import pack, verified_archive
from enter_eval_container import enter
from balance_evaluator_runtime import balance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--songeval', type=Path, required=True)
    p.add_argument('--python', required=True)
    p.add_argument('--enable-q3o', action='store_true', help='Deferred unless explicitly authorized')
    a = p.parse_args()
    enter(a.work)
    balance(a.work)
    clean_install_cache(a.work)
    import torch
    if torch.cuda.get_device_properties(0).total_memory < 79 * (1 << 30):
        raise RuntimeError('An 80GB-class GPU is required')
    code = Path(__file__).parent
    bundle = a.work / 'wsb_metrics'
    generation_source = json.loads((a.runs / 'generation-source.json').read_text())
    for arm in ARMS:
        run = a.runs / f'yue2-{arm}' / 'wsb'
        chunk = run / 'evaluation/chunk_000_004.json'
        if verified_archive(run, chunk):
            continue
        if not chunk.exists():
            subprocess.run([a.python, '-u', str(code / 'score_objective.py'), '--bundle', str(bundle),
                '--run', str(run), '--source', str(code / 'reproduction_manifest.jsonl'),
                '--start', '0', '--end', '4', '--extras-python', str(a.work / 'extras/bin/python'),
                '--songeval', str(a.songeval)], check=True)
        data = json.loads(chunk.read_text())
        exported = next(r['generation_receipts'] for r in generation_source['datasets']['wsb']['rows']
                        if r['id'] == f'yue2-{arm}')
        by_id = {r['id']: r for r in exported}
        if len(by_id) != 8:
            raise ValueError('Expected both original candidates for four conditions')
        for row in data['rows']:
            audio = chunk.parent / 'audio' / row['id'] / 'audio.mp3'
            if sha(audio) != row['audio_sha256']:
                raise ValueError('Staged selected MP3 hash changed')
            binding = by_id[row['id']]
            if binding['audio_sha256'] != row['source_flac_sha256'] or binding['adapter_sha256'] != row['adapter_sha256']:
                raise ValueError('Original generation receipt does not bind the selected candidate')
            row['audio_path'] = str(audio.resolve())
        write(run / 'generation-receipts.json', exported)
        write(chunk, data)
        if a.enable_q3o:
            subprocess.run([a.python, '-u', str(code / 'score_q3o.py'), '--bundle', str(bundle),
                            '--chunk', str(chunk)], check=True)
        pack(run, chunk, require_q3o=a.enable_q3o)
    write(a.runs / 'completion.json', {'status': 'complete', 'conditions': [0, 4],
                                       'arms': ARMS, 'new_training': False})


if __name__ == '__main__':
    main()
