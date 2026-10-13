"""Finite full-cohort CMI transport/submission stage; no scheduled recovery."""

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from generate_objective import write

LAB = '/storage/zhanh_storage/lyrics-objective-eval-20261013'
KAYA = '/scratch/ems011/zhe/wsb-objective-20261013'
ARMS = ['yue2-baseline', 'yue2-songeval-2e-5', 'yue2-musecritic-2e-5',
        'yue2-songeval-1e-4', 'yue2-musecritic-1e-4',
        'yue2-songeval-3e-4', 'yue2-musecritic-3e-4']


def assignments():
    return [(arm, start, min(start+4, 121)) for start in range(4, 121, 4) for arm in ARMS]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--controller-pid', type=int, required=True)
    p.add_argument('--timeout-seconds', type=int, default=72*3600)
    a = p.parse_args()
    receipt = json.loads(a.receipt.read_text()) if a.receipt.exists() else {
        'scope': 'one full CMI test campaign, 121 conditions x 7 arms x 2 candidates',
        'prefix_conditions_verified': [0, 4], 'remaining_candidates': 1638,
        'expected_assignments': 210, 'scheduled_task': False, 'new_training': False,
        'automatic_result_recovery': False, 'assignments': [], 'status': 'running'}
    done = {(r['arm'], r['conditions'][0], r['conditions'][1]) for r in receipt['assignments']}
    deadline = time.monotonic() + a.timeout_seconds

    def run(command):
        return subprocess.check_output(command, text=True, timeout=900).strip()

    try:
        for arm, start, end in assignments():
            if (arm, start, end) in done:
                continue
            key = f'{arm}-{start:03d}-{end:03d}'
            source = f'{LAB}/runs/{arm}/cmi/evaluation/chunk_{start:03d}_{end:03d}.json'
            receipt.update(status='running', waiting_for=key, updated_at=datetime.now(timezone.utc).isoformat())
            write(a.receipt, receipt)
            while True:
                ready = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'lab5090', 'test', '-f', source], timeout=30)
                if ready.returncode == 0:
                    break
                if ready.returncode != 1:
                    raise RuntimeError('Generation host is unreachable, not an incomplete chunk')
                if time.monotonic() > deadline:
                    raise TimeoutError('Finite campaign transport deadline exceeded')
                alive = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'lab5090',
                                        'kill', '-0', str(a.controller_pid)], timeout=30)
                if alive.returncode != 0:
                    raise RuntimeError('Full generator stopped before preparing all CMI inputs')
                time.sleep(30)
            run(['ssh', 'lab5090', '/home/mengh/research/YuE2-posttrain/.venv/bin/python',
                 '/home/mengh/research/WildSongBench-eval/code/export_quality_inputs.py',
                 '--run', f'{LAB}/runs/{arm}/cmi', '--output', f'{LAB}/kaya-full-staging/{key}',
                 '--start', str(start), '--end', str(end)])
            input_root = f'{KAYA}/cmi-full-input/{key}'
            output_root = f'{KAYA}/cmi-full-quality/{key}'
            run(['ssh', 'kaya', 'mkdir', '-p', input_root])
            run(['scp', '-3', '-r', f'lab5090:{LAB}/kaya-full-staging/{key}/{arm}', f'kaya:{input_root}/'])
            jobname = 'cmi-full-' + key
            # On restart, resolve an unknown submission outcome by the unique
            # campaign/chunk name rather than silently submitting duplicate work.
            previous = run(['ssh', 'kaya', 'sacct', '-n', '-X', '-S', '2026-10-13',
                            '--name', jobname, '--format=JobIDRaw', '--parsable2']).splitlines()
            ids = [line.strip().rstrip('|') for line in previous if line.strip()]
            if len(ids) > 1:
                raise ValueError('Duplicate Slurm submissions require manual inspection: ' + key)
            if ids:
                job = ids[0]
            else:
                receipt.update(submission_in_flight=key)
                write(a.receipt, receipt)
                job = run(['ssh', 'kaya', 'sbatch', '--parsable', f'--job-name={jobname}',
                    f'--export=ALL,CMI_ARM={arm},CMI_INPUT={input_root},CMI_OUTPUT={output_root}',
                    f'--output={KAYA}/logs/{key}-%j.out', f'--error={KAYA}/logs/{key}-%j.err',
                    f'{KAYA}/code/kaya_cmi_quality.sbatch']).split(';')[0]
            if not job.isdigit():
                raise ValueError('Unrecognized Slurm submission receipt')
            record = {'arm': arm, 'job': job, 'conditions': [start, end], 'candidates': 2*(end-start),
                      'input_root': input_root, 'output_root': output_root,
                      'submitted_at': datetime.now(timezone.utc).isoformat()}
            receipt['assignments'].append(record)
            receipt.pop('submission_in_flight', None)
            write(a.receipt, receipt)
            print(json.dumps(record), flush=True)
        receipt.update(status='all-full-cohort-inputs-submitted', waiting_for=None)
    except Exception as error:
        receipt.update(status='failed', failure=str(error))
        raise
    finally:
        receipt['updated_at'] = datetime.now(timezone.utc).isoformat()
        write(a.receipt, receipt)


if __name__ == '__main__':
    main()
