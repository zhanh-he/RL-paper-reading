"""Publish verified Muse training traces without inventing validation or replays."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summaries', type=Path, required=True)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--registry', type=Path, required=True)
    p.add_argument('--resume-job', required=True)
    a = p.parse_args()
    data = json.loads(a.registry.read_text())
    a.archive.mkdir(parents=True, exist_ok=True)
    for reward in ['songeval', 'musecritic']:
        for lr in ['2e-5', '1e-4', '3e-4']:
            summary = json.loads((a.summaries / f'recovered_{reward}_{lr}.json').read_text())
            ident = f'muse-{reward}-{lr}'
            if summary['status'] not in ['verified-training-complete', 'verified-partial-failed']:
                raise ValueError('Unverified summary')
            if len(summary['train_curve']) != summary['verified_train_steps'] or summary['validation']:
                raise ValueError('Inconsistent training-only summary')
            arm = next(r for r in data['arms'] if r['id'] == ident)
            arm.update({key: summary[key] for key in ['status', 'verified_train_steps', 'train_curve', 'checkpoints', 'failure_reason', 'validation', 'invalid_output_count']})
            arm['training_receipt'] = f'../../../music-gen/lyrics2song/rl/grpo/runs/2026-10-14-formal-recovery/{ident}.json'
            if reward == 'musecritic' and lr == '3e-4':
                arm['resume'] = {'job_id': a.resume_job, 'project': 'iv96', 'priority': 1023,
                                 'from_step': 180, 'target_step': 270, 'status': 'submitted-not-completed',
                                 'restore_optimizer_scheduler_rng': True}
            (a.archive / f'{ident}.json').write_text(json.dumps(summary, indent=2) + '\n')
    data['recovered_at'] = '2026-10-14'
    a.registry.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
