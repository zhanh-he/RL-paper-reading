"""Recover same-candidate CMI quality chunks, never publish partial means."""

import argparse
import json
from pathlib import Path

from generate_objective import sha, write
from merge_returned_quality import merge


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--returned', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    a = p.parse_args()
    recovered = []
    for returned in sorted(a.returned.glob('*/*/cmi/evaluation/chunk_*.json')):
        data = json.loads(returned.read_text())
        if not data.get('complete') or not all('supplementary' in row for row in data['rows']):
            continue
        arm = returned.parents[2].name
        original = a.runs / arm / 'cmi/evaluation' / returned.name
        source = json.loads(original.read_text())
        digest = sha(returned)
        if source.get('remote_recovery', {}).get('returned_chunk_sha256') == digest:
            result = source
        else:
            result = merge(source, data)
            for row in result['rows']:
                if sha(Path(row['audio_path'])) != row['audio_sha256']:
                    raise ValueError('Original scoring MP3 changed')
            result['remote_recovery'] = {'returned_chunk_sha256': digest, 'source_chunk_sha256': sha(original),
                'scope': 'same immutable candidates; remote scoring, no redraw'}
            write(original, result)
        recovered.append({'arm': arm, 'chunk': returned.name, 'candidates': len(result['rows']),
                          'returned_chunk_sha256': digest, 'merged_chunk_sha256': sha(original)})
    write(a.receipt, {'scope': 'partial full-cohort progress, not final means',
                     'chunks': recovered, 'candidate_count': sum(r['candidates'] for r in recovered)})
    print(len(recovered), 'chunks,', sum(r['candidates'] for r in recovered), 'candidates recovered')


if __name__ == '__main__':
    main()
