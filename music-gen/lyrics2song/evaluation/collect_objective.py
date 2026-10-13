"""Publish aggregate receipts only; never copy benchmark prompts or private audio."""

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from generate_objective import sha, write


METRICS = ['songbench_global_avg', 'songbench_musicality', 'songeval_global_avg',
           'songeval_musicality', 'audiobox_pq', 'mulan', 'allmusiccaps', 'control_overall', 'per']


def cell(key, dataset, q3o_submitted=False):
    if dataset == 'cmi' and key in ['per', 'control_overall']:
        return {'status': 'not-applicable', 'value': None, 'scored': 0, 'reason':
            '全体 test 含越南语、爪哇语等；官方中英 G2P 未验证适用，不能报全体 PER。' if key == 'per' else
            'CMI-Pref 没有 WSB 的冻结逐条件 Q3O 属性/权重；不能套用另一个数据集的标注。'}
    return {'status': 'resource-pending' if key == 'control_overall' and not q3o_submitted else 'pending', 'value': None,
            'scored': 0, 'reason': ('Q3O 已提交 Gadi 80GB GPU，尚未收齐评分。' if q3o_submitted else
            'Q3O 官方需 80 GB；5090 为 32 GB，尚未提交该 scorer。') if key == 'control_overall' else '尚未完成完整测试集。'}


def aggregate(run, dataset, expected, q3o_submitted=False):
    runs = [run] if isinstance(run, Path) else run
    cells = {key: cell(key, dataset, q3o_submitted) for key in METRICS}
    chunks = [f for root in runs for f in sorted((root / 'evaluation').glob('chunk_*.json'))]
    rows = [r for f in chunks for r in json.loads(f.read_text())['rows']]
    if len({r.get('adapter_sha256') for r in rows}) > 1:
        raise ValueError('Checkpoint identity differs across shards')
    counts = Counter(r['prompt_index'] for r in rows)
    if any(not 0 <= i < expected or n != (1 if dataset == 'wsb' else 2) for i, n in counts.items()):
        raise ValueError('Wrong selected-candidate coverage')
    if dataset == 'cmi' and any({r['candidate_index'] for r in rows if r['prompt_index'] == i} != {0, 1} for i in counts):
        raise ValueError('CMI must retain both candidates per condition')
    supplemental = [r['supplementary'] for r in rows if 'supplementary' in r]
    if len({json.dumps(s['scorer'], sort_keys=True) for s in supplemental}) > 1 or len({s['audiobox_state_sha256'] for s in supplemental}) > 1:
        raise ValueError(f'Supplementary scorer identity drift: {run}')
    ids = [r['id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f'Overlapping evaluated chunks: {run}')
    values = {key: [] for key in METRICS}
    diagnostic_rows = []
    edits = phonemes = 0
    for row in rows:
        scores = row['metrics']
        raw = scores.get('songbench', {}).get('songbench_raw')
        current = {key: None for key in METRICS}
        if raw is not None:
            current['songbench_global_avg'] = sum(raw.values()) / 7
            current['songbench_musicality'] = raw['Musicality']
        for key in ['mulan', 'allmusiccaps']:
            if key in scores:
                current[key] = scores[key]['scores'][key]
        if dataset == 'wsb' and 'per' in scores:
            edits += scores['per']['edits']; phonemes += scores['per']['reference_phonemes']
            current['per'] = scores['per']['per']
        if dataset == 'wsb' and 'q3o' in scores and scores['q3o']['status'] == 'complete':
            current['control_overall'] = scores['q3o']['scores']['q3o']
        if 'supplementary' in row:
            se = row['supplementary']['songeval']
            current['songeval_global_avg'] = sum(se.values()) / 5
            current['songeval_musicality'] = se['Musicality']
            current['audiobox_pq'] = row['supplementary']['audiobox']['PQ']
        for key, value in current.items():
            if value is not None:
                values[key].append(value)
        diagnostic_rows.append({'id': row['id'], 'condition_index': row['prompt_index'],
            'candidate_index': row['candidate_index'], 'seconds': row['seconds'],
            'scoring_mp3_sha256': row['audio_sha256'], 'source_flac_sha256': row['source_flac_sha256'],
            'songbench_dimensions': raw, 'metrics': current,
            'scope': 'individual candidate diagnostic, not full-cohort metric; PER here is per-song, not corpus micro'})
    target_count = expected if dataset == 'wsb' else expected * 2
    for key, items in values.items():
        if items:
            if len(items) > target_count:
                raise ValueError('More test scores than expected')
            cells[key].update(scored=len(items), expected=target_count)
            if len(items) == target_count:
                cells[key].update(status='complete', value=edits / phonemes if key == 'per' else sum(items) / len(items), reason='完整测试集实测。')
    generation = []
    for root in runs:
        per_root = {r['id']: r for r in [json.loads(f.read_text()) for f in
                    sorted((root / 'songs').glob('*/evaluation_binding.json'))]}
        exported = root / 'generation-receipts.json'
        if exported.is_file():
            for r in json.loads(exported.read_text()):
                if r['id'] in per_root and per_root[r['id']] != r:
                    raise ValueError('Archived/live generation receipt drift')
                per_root[r['id']] = r
        generation.extend(per_root.values())
    if len({r['id'] for r in generation}) != len(generation):
        raise ValueError('Duplicate generation receipts across shard roots')
    generated = len(generation)
    return {'generated_candidates': generated, 'expected_candidates': expected * 2,
            'scored_conditions': len({r['prompt_index'] for r in rows if 'songbench' in r['metrics']}), 'metrics': cells,
            'diagnostic_rows': diagnostic_rows,
            'generation_receipts': generation,
            'receipt_files': [{'file': str(f), 'sha256': sha(f)} for f in chunks]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, action='append', required=True, help='Repeat for disjoint shard roots; do not include numerical-audit duplicates')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--queue-pid', type=int)
    p.add_argument('--execution', type=Path, help='Verified submission receipt (not a scheduler)')
    a = p.parse_args()
    alive = False
    if a.queue_pid:
        try:
            os.kill(a.queue_pid, 0)
            alive = True
        except ProcessLookupError:
            pass
    arms = [('yue2-baseline', 'YuE2-Baseline', 'scheduled')]
    for reward in ['songeval', 'musecritic']:
        for lr in ['2e-5', '1e-4', '3e-4']:
            arms.append((f'yue2-{reward}-{lr}', f'YuE2-{ "SongEval" if reward == "songeval" else "MuseCritic"}-{lr}', 'scheduled'))
    arms.append(('muse-baseline', 'Muse-Baseline', 'not-submitted'))
    for reward in ['SongEval', 'MuseCritic']:
        for lr in ['2e-5', '1e-4', '3e-4']:
            arms.append((f'muse-{reward.lower()}-{lr}', f'Muse-{reward}-{lr}', 'waiting-for-training'))
    data = {'schema': 1, 'updated_at': datetime.now(timezone.utc).isoformat(),
            'host': 'lab5090', 'queue_pid': a.queue_pid, 'metrics': METRICS,
            'queue_running_at_collection': alive,
            'test_authorized_at': '2026-10-13', 'datasets': {}}
    if a.execution:
        data['execution'] = json.loads(a.execution.read_text())
    for dataset, count in [('wsb', 192), ('cmi', 121)]:
        data['datasets'][dataset] = {'conditions': count, 'rows': []}
        for ident, label, status in arms:
            result = aggregate([r / ident / dataset for r in a.runs], dataset, count, bool(a.execution) and ident.startswith('yue2-'))
            data['datasets'][dataset]['rows'].append({'id': ident, 'system': label, 'step': 0 if ident.endswith('baseline') else 270,
                'status': 'evaluating' if result['generated_candidates'] else status, **result})
    write(a.output, data)


if __name__ == '__main__':
    main()
