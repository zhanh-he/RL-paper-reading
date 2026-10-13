"""Export new CMI quality inputs, binding every MP3 to its generation receipt."""

import argparse
import json
import shutil
from pathlib import Path

from generate_objective import sha, write


def validate_assignment(chunk):
    if chunk['generation_protocol']['dataset'] != 'cmi' or len(chunk['rows']) != 8:
        raise ValueError('Expected both candidates of the four reserved CMI conditions')
    expected = {f'cmi_{i:03d}_r{j}_s{[931001, 931019][j]+i}' for i in range(4) for j in range(2)}
    if {r['id'] for r in chunk['rows']} != expected:
        raise ValueError('Reserved CMI candidate coverage changed')
    for row in chunk['rows']:
        if (row['prompt_index'] not in range(4) or row['candidate_index'] not in (0, 1)
                or row['seed'] != [931001, 931019][row['candidate_index']] + row['prompt_index']
                or row['id'] != f"cmi_{row['prompt_index']:03d}_r{row['candidate_index']}_s{row['seed']}"):
            raise ValueError('Reserved CMI ID/seed binding changed')
    if any(row['metrics'] for row in chunk['rows']):
        raise ValueError('Already-scored rows are audits, not new production assignments')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    source = a.run / 'evaluation/chunk_000_004.json'
    chunk = json.loads(source.read_text())
    validate_assignment(chunk)
    arm = a.run.parent.name
    output = a.output / arm / 'cmi'
    generation = []
    for row in chunk['rows']:
        if row['prompt_index'] not in range(4):
            raise ValueError('Kaya assignment overlaps a Gadi shard')
        binding = json.loads((a.run / 'songs' / row['id'] / 'evaluation_binding.json').read_text())
        if (binding['adapter_sha256'] != row['adapter_sha256']
                or binding['audio_sha256'] != row['source_flac_sha256']
                or binding['seed'] != row['seed']
                or binding['source_sha256'] != chunk['generation_protocol']['source_sha256']
                or binding['condition_index'] != row['prompt_index']
                or binding['candidate_index'] != row['candidate_index']):
            raise ValueError('Generation-to-MP3 binding changed')
        audio = source.parent / 'audio' / row['id'] / 'audio.mp3'
        if sha(audio) != row['audio_sha256']:
            raise ValueError('Scoring MP3 changed before transfer')
        target = output / 'evaluation/audio' / row['id'] / 'audio.mp3'
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(audio, target)
        if sha(target) != row['audio_sha256']:
            raise ValueError('Exported MP3 changed')
        generation.append(binding)
    write(output / 'evaluation' / source.name, chunk)
    write(output / 'generation-receipts.json', generation)
    write(output / 'quality-assignment.json', {'dataset': 'cmi', 'conditions': [0, 4],
        'candidates': 8, 'arm': arm, 'source_chunk_sha256': sha(source),
        'selection': 'none; both fixed seeds', 'new_training': False})


if __name__ == '__main__':
    main()
