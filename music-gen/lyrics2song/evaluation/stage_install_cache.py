"""Move this campaign's UV cache to allocation-local storage, not shared inodes."""

import argparse
import json
import shutil
from pathlib import Path


def stage(bundle, jobfs, key_root):
    cache = bundle / '.cache/uv'
    target = jobfs / 'uv'
    marker = bundle.parent / 'installation-cache.json'
    if cache.is_symlink():
        previous = json.loads(marker.read_text())
        if str(cache.readlink()) != previous['target']:
            raise ValueError('Installation cache link has an unknown owner')
        cache.unlink()
    if target.exists():
        raise ValueError('Refuse to overwrite an allocation-local cache')
    jobfs.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        # UV_LINK_MODE=copy leaves the installed environments independent.
        shutil.move(str(cache), str(target))
    else:
        target.mkdir()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.symlink_to(target)
    marker.write_text(json.dumps({'target': str(target), 'temporary': True}) + '\n')
    key_root.mkdir(parents=True, exist_ok=True)
    key = bundle / '.venvs/key'
    if key.is_symlink():
        if key.resolve() != key_root.resolve():
            raise ValueError('Key environment already points to another allocation')
    elif key.exists():
        raise ValueError('Refuse to replace an existing key environment directory')
    else:
        key.symlink_to(key_root)


def finish(bundle):
    receipt = json.loads((bundle.parent / 'preparation-receipt.json').read_text())
    if receipt['status'] != 'prepared':
        raise ValueError('Preparation must finish before releasing the cache link')
    cache = bundle / '.cache/uv'
    marker = json.loads((bundle.parent / 'installation-cache.json').read_text())
    if not cache.is_symlink() or str(cache.readlink()) != marker['target']:
        raise ValueError('Installation cache link changed')
    cache.unlink()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--jobfs', type=Path)
    p.add_argument('--key-root', type=Path)
    p.add_argument('--finish', action='store_true')
    a = p.parse_args()
    if a.finish:
        finish(a.bundle)
    else:
        if a.jobfs is None or a.key_root is None:
            p.error('--jobfs and --key-root are required when staging')
        stage(a.bundle, a.jobfs, a.key_root)
