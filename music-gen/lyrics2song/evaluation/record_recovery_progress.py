"""Bind a manual result-recovery snapshot to published execution provenance."""
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execution', type=Path, required=True)
    p.add_argument('--dispatcher', type=Path, required=True)
    p.add_argument('--quality-recovery', type=Path, required=True)
    p.add_argument('--registry', type=Path, required=True)
    a = p.parse_args()
    data = json.loads(a.execution.read_text())
    transport = json.loads(a.dispatcher.read_text())
    recovery = json.loads(a.quality_recovery.read_text())
    registry = json.loads(a.registry.read_text())
    data['q3o_deferred'] = True
    data['q3o_policy'] = 'User deferred 2026-10-14; weights preserved; no Q3O worker launched by existing queues.'
    data['manual_recovery_at'] = datetime.now(timezone.utc).isoformat()
    data['kaya']['full_cmi_campaign'].update({
        'submitted_chunks_at_recovery': len(transport['assignments']),
        'submitted_candidates_at_recovery': sum(r['candidates'] for r in transport['assignments']),
        'recovered_chunks': len(recovery['chunks']),
        'recovered_candidates': recovery['candidate_count'],
        'completed_prefix_candidates': 56,
        'waiting_for_generation': transport['waiting_for']})
    data['muse_training'] = [{key: r.get(key) for key in
        ['id', 'status', 'verified_train_steps', 'checkpoints', 'resume', 'invalid_output_count']}
        for r in registry['arms'] if r['model'] == 'Muse']
    data['gadi']['queue_snapshot'] = {'wa66_evaluation': {'queued': 6, 'running': 0},
        'iv96_evaluation': {'queued': 7, 'running': 0},
        'priority': 1023, 'reason': 'Insufficient ngpus', 'scheduler_start_estimate': None,
        'shared_pool': True, 'queue_started_awst': '2026-10-13T13:55:47+08:00',
        'scope': 'manually verified snapshot, not live queue state'}
    a.execution.write_text(json.dumps(data, indent=2) + '\n')


if __name__ == '__main__':
    main()
