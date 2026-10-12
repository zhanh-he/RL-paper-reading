"""Submit six fresh formal Muse arms; no independent inference or validation."""

import datetime
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path('/g/data/wa66/hanyu/vocal2accomp-muse')
STUDY = ROOT / 'experiments/cmi_aligned_lr_20261012'
SOURCE = ROOT / 'experiments/cmi_triple_9to1_20261006'
HASHES = {'muse-cmi-triple-train.jsonl': 'ff437d751d62742d92daf7ac56a678010c52d6c22def5624a8324301e6699be6',
          'muse-cmi-triple-valid.jsonl': 'ed5c5f36af48ef6ce81d15c7b15b429fb250ec4a52a04a43e55d96ec3cafb59d'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replace-queued', action='store_true')
    args = parser.parse_args()
    (STUDY / 'data').mkdir(exist_ok=True)
    (STUDY / 'pbs_logs').mkdir(exist_ok=True)
    for name, expected in HASHES.items():
        source = SOURCE / 'data' / name
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected, name
        destination = STUDY / 'data' / name
        if not destination.exists(): destination.symlink_to(source)
        assert hashlib.sha256(destination.read_bytes()).hexdigest() == expected
    for reward in ('songeval', 'musecritic'):
        for lr in ('2e-5', '1e-4', '3e-4'):
            receipt = STUDY / f'submission_{reward}_lr{lr}.json'
            if receipt.exists():
                if not args.replace_queued:
                    print(receipt.read_text().strip())
                    continue
                previous = json.loads(receipt.read_text())
                status = subprocess.check_output(['qstat', '-f', previous['job_id']], universal_newlines=True)
                if 'job_state = Q' not in status:
                    raise RuntimeError('Refusing automatic replacement of nonqueued job')
                subprocess.check_call(['qdel', previous['job_id']])
                history = STUDY / 'superseded_submissions'
                history.mkdir(exist_ok=True)
                receipt.rename(history / f'{receipt.stem}_{previous["job_id"]}.json')
            project = 'wa66' if reward == 'songeval' else 'iv96'
            variables = ','.join([f'STUDY={STUDY}', 'ALIGNED_LR=1',
                                  f'MUSE_REWARD_PLUGIN={STUDY}/muse_cmi_aligned_rewards.py',
                                  'MODE=train', f'REWARD={reward}', f'LR={lr}'])
            command = ['qsub', '-P', project, '-p', '1023', '-N', f'muse-{reward[:2]}-{lr}',
                       '-o', str(STUDY / f'pbs_logs/{reward}_{lr}.out'),
                       '-e', str(STUDY / f'pbs_logs/{reward}_{lr}.err'),
                       '-v', variables, str(STUDY / 'gadi_muse_cmi_triple_baseline.pbs')]
            job_id = subprocess.check_output(command, cwd=STUDY, universal_newlines=True).strip()
            if not job_id.endswith('.gadi-pbs'): raise RuntimeError(f'Unexpected submission: {job_id}')
            record = {'id': f'muse-{reward}-{lr}', 'model': 'Muse',
                      'reward': 'SongEval' if reward == 'songeval' else 'MuseCritic',
                      'lr': lr, 'host': 'gadi', 'project': project, 'job_id': job_id,
                      'priority': 1023, 'submitted_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      'status': 'queued', 'verified_train_steps': 0, 'planned_steps': 270,
                      'train_curve': [], 'checkpoints': [], 'validation': {}, 'long_replays': {},
                      'train_conditions': 270, 'valid_conditions': 30, 'sealed_test_conditions': 121,
                      'fresh_base': True, 'beta': .01, 'seed': 42,
                      'standalone_inference': False, 'validation_generation': False,
                      'invalid_output_policy': 'zero for missing AUDIO, nonfinite or under 1-second decoded audio; infrastructure errors fatal',
                      'command': command,
                      'scripts_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                         [STUDY / 'gadi_muse_cmi_triple_baseline.pbs', STUDY / 'muse_cmi_rewards.py', STUDY / 'muse_cmi_aligned_rewards.py']}}
            receipt.write_text(json.dumps(record, indent=2) + '\n')
            print(json.dumps(record))


if __name__ == '__main__': main()
