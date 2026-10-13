"""Preserve failed receipts while allowing a finite, explicit scoring retry."""

import shutil

from generate_objective import sha


def needs_scoring(path, cached):
    if cached and cached.get('status') == 'complete':
        return False
    if path.exists():
        archive = path.parent / 'failed-attempts' / f'{path.stem}.{sha(path)}.json'
        archive.parent.mkdir(exist_ok=True)
        if not archive.exists():
            shutil.copy2(path, archive)
    return True
