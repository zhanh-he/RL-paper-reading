"""Finite transfer/submission stage of the running evaluation, never a scheduler."""

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from generate_objective import write

ARMS = ['yue2-musecritic-1e-4', 'yue2-songeval-3e-4', 'yue2-musecritic-3e-4']
LAB = '/storage/zhanh_storage/lyrics-objective-eval-20261013'
KAYA = '/scratch/ems011/zhe/wsb-objective-20261013'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--controller-pid', type=int, required=True)
    p.add_argument('--timeout-seconds', type=int, default=14400)
    a = p.parse_args()
    deadline = time.monotonic() + a.timeout_seconds
    receipt = {'scope': 'one finite evaluation data-transfer pipeline; no periodic recovery or notification',
               'scheduled_task': False, 'new_training': False, 'assignments': [], 'status': 'running'}
    if a.receipt.exists():
        receipt = json.loads(a.receipt.read_text())
    completed = {row['arm'] for row in receipt['assignments']}

    def run(command):
        return subprocess.check_output(command, text=True, timeout=600).strip()

    try:
        for arm in ARMS:
            if arm in completed:
                continue
            source = f'{LAB}/runs/{arm}/cmi/evaluation/chunk_000_004.json'
            receipt.update(waiting_for=arm, updated_at=datetime.now(timezone.utc).isoformat())
            write(a.receipt, receipt)
            while True:
                ready = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'lab5090', 'test', '-f', source], timeout=30)
                if ready.returncode == 0:
                    break
                if ready.returncode != 1:
                    raise RuntimeError('Source host failed, not an incomplete generation chunk')
                if time.monotonic() > deadline:
                    raise TimeoutError('Finite evaluation transfer deadline exceeded')
                alive = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'lab5090',
                                        'kill', '-0', str(a.controller_pid)], timeout=30)
                if alive.returncode != 0:
                    raise RuntimeError('Generation controller exited before completing its reserved chunks')
                time.sleep(30)
            run(['ssh', 'lab5090', '/home/mengh/research/YuE2-posttrain/.venv/bin/python',
                 '/home/mengh/research/WildSongBench-eval/code/export_quality_inputs.py',
                 '--run', f'{LAB}/runs/{arm}/cmi', '--output', f'{LAB}/kaya-staging/{arm}'])
            run(['ssh', 'kaya', 'mkdir', '-p', f'{KAYA}/cmi-input/{arm}'])
            run(['scp', '-3', '-r', f'lab5090:{LAB}/kaya-staging/{arm}/{arm}', f'kaya:{KAYA}/cmi-input/{arm}/'])
            job = run(['ssh', 'kaya', 'sbatch', '--parsable', f'--export=ALL,CMI_ARM={arm}',
                       f'--output={KAYA}/logs/cmi-{arm}.out', f'--error={KAYA}/logs/cmi-{arm}.err',
                       f'{KAYA}/code/kaya_cmi_quality.sbatch'])
            if not job.split(';')[0].isdigit():
                raise ValueError('Unrecognized Slurm submission receipt')
            receipt['assignments'].append({'arm': arm, 'job': job.split(';')[0],
                'conditions': [0, 4], 'candidates': 8, 'submitted_at': datetime.now(timezone.utc).isoformat()})
            write(a.receipt, receipt)
            print(json.dumps(receipt['assignments'][-1]), flush=True)
        receipt.update(status='all-reserved-inputs-submitted', waiting_for=None)
    except Exception as error:
        receipt.update(status='failed', failure=str(error))
        raise
    finally:
        receipt['updated_at'] = datetime.now(timezone.utc).isoformat()
        write(a.receipt, receipt)


if __name__ == '__main__':
    main()
