"""Parallel pinned public-asset staging; the official preparer remains intact."""

import argparse
import concurrent.futures
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

from generate_objective import sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    inventory = a.bundle / 'assets.lock.json'
    rows = [r for r in json.loads(inventory.read_text())['files']
            if 'q3o' in r['metrics'] and 'repo_id' in r and r['bytes'] > (100 << 20)]

    def download(row):
        path = a.bundle / row['path']
        if path.is_file() and path.stat().st_size == row['bytes'] and sha(path) == row['sha256']:
            return {'path': row['path'], 'status': 'already-verified'}
        # Separate temporary names make a race with the original serial preparer
        # safe: both writers may only replace the target with hash-identical bytes.
        if path.with_name(path.name + '.download').exists():
            return {'path': row['path'], 'status': 'owned-by-official-downloader'}
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + '.parallel-' + str(os.getpid()))
        url = 'https://huggingface.co/' + row['repo_id'] + '/resolve/' + row['revision'] + '/' + urllib.parse.quote(row['filename'])
        for attempt in range(3):
            try:
                request = urllib.request.Request(url, headers={'User-Agent': 'wsb-pinned-prestage/1'})
                with urllib.request.urlopen(request, timeout=120) as response, temp.open('wb') as out:
                    while block := response.read(8 << 20):
                        out.write(block)
                if temp.stat().st_size != row['bytes'] or sha(temp) != row['sha256']:
                    raise ValueError('Frozen asset size/hash mismatch: ' + row['path'])
                temp.replace(path)
                print('Verified ' + row['path'], flush=True)
                return {'path': row['path'], 'status': 'downloaded-and-verified'}
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)

    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        receipts = list(pool.map(download, rows))
    write(a.bundle.parent / 'parallel-prestage-receipt.json', {'status': 'complete',
        'inventory_sha256': sha(inventory), 'code_sha256': sha(Path(__file__)),
        'workers': a.workers, 'files': receipts, 'new_training': False})


if __name__ == '__main__':
    main()
