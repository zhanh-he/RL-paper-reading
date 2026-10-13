"""Relocate the completed eval venv without changing its logical prefix."""

import argparse
import fcntl
import json
import shutil

from pathlib import Path


def balance(work, installer_paused=False):
    if not installer_paused and json.loads((work / 'preparation-receipt.json').read_text())['status'] != 'prepared':
        raise ValueError('Do not move an installing environment')
    source = work / 'wsb_metrics/.venvs/eval'
    target = work / 'evaluator-eval'
    with (work / 'runtime-relocation.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if source.is_symlink():
            if source.resolve() != target.resolve():
                raise ValueError('Eval runtime has an unknown target')
            return
        if target.exists():
            raise ValueError('An incomplete prior runtime relocation needs inspection')
        if not source.is_dir():
            raise FileNotFoundError('Eval runtime was not installed')
        shutil.move(str(source), str(target))
        source.symlink_to(target)
        (work / 'runtime-relocation.json').write_text(json.dumps({
            'logical_prefix': str(source), 'physical_target': str(target),
            'installed_packages_changed': False}) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--installer-paused', action='store_true')
    a = p.parse_args()
    balance(a.work, a.installer_paused)
