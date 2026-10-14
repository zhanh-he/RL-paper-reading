"""Export attributed CMI inputs and two explicitly modified listening probes."""
import argparse
import hashlib
import json
from pathlib import Path

DATA_SHA = 'c39d43e81793bee3f312c7028a0483b5da41068db4544acd258115043101e46f'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if hashlib.sha256(a.data.read_bytes()).hexdigest() != DATA_SHA:
        raise ValueError('Frozen projected dataset mismatch')
    rows = json.loads(a.data.read_text())['valid']
    assert len(rows) == 30
    inputs = [{'index': i, **row, 'seed': 5101+i, 'base_valid_index': i,
               'kind': 'original-validation', 'requested_count': None,
               'scope': 'Unmodified CMI validation condition; no exact instrument-count constraint'}
              for i, row in enumerate(rows)]
    for index, families in [(30, ['drums', 'electric bass']),
                            (31, ['drums', 'electric bass', 'piano', 'electric guitar', 'strings'])]:
        row = rows[0]
        instruction = (f'Use exactly {len(families)} accompaniment instrument families, excluding the singing voice: '
                       + ', '.join(families) + '. Do not add any other accompaniment instrument families.')
        inputs.append({'index': index, **row, 'style': row['style']+'\n'+instruction,
                       'original_style': row['style'], 'seed': 5101, 'base_valid_index': 0,
                       'kind': 'instrument-count-probe', 'requested_count': len(families),
                       'instrument_families': families, 'adaptation': instruction,
                       'scope': 'Modified listening probe only; excluded from train, all-30 validation and sealed test'})
    report = {'schema': 1, 'data_sha256': DATA_SHA, 'display_indices': [0, 10, 20, 30, 31],
              'display_indices_by_model': {'songbloom': [0, 10, 20, 1, 2]},
              'selection': 'Three original examples retained; paired count-2/count-5 probes reuse valid0 lyrics and seed5101; no score-based selection. SongBloom uses five distinct native conditions0/10/20/1/2 because it does not consume Text.',
              'attribution': {'dataset': 'CMI-Pref', 'authors': 'Haiwen Xia et al.',
                              'url': 'https://huggingface.co/datasets/HaiwenXia/cmi-pref',
                              'revision': '5282fbe784e326014894b299cb22645cd7d56057',
                              'license': 'CC BY-NC-SA 4.0',
                              'license_url': 'https://creativecommons.org/licenses/by-nc-sa/4.0/',
                              'changes': 'Original text/lyrics unchanged in0..29; style instructions appended only in30/31. No reference audio redistributed.'},
              'inputs': inputs}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'display_indices': report['display_indices'], 'original_rows': len(rows), 'probes': 2}))


if __name__ == '__main__':
    main()
