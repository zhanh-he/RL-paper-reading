"""Upload missing immutable FLAC release assets and validate server digests."""

import argparse
import hashlib
import json
import subprocess


def inventory():
    raw = subprocess.check_output(['gh', 'api', 'repos/zhanh-he/RL-paper-reading/releases/tags/formal-cmi-pref-20261012'])
    return {row['name']: row for row in json.loads(raw)['assets']}


def main():
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    wanted = json.loads((args.folder / 'asset-bindings.json').read_text())
    existing = inventory()
    missing = []
    for name, binding in wanted.items():
        path = args.folder / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != binding['sha256']:
            raise ValueError(f'Local asset changed: {name}')
        if name in existing:
            if existing[name].get('digest') != 'sha256:' + binding['sha256']:
                raise ValueError(f'Release digest mismatch: {name}')
        else:
            missing.append(str(path))
    for start in range(0, len(missing), 10):
        subprocess.run(['gh', 'release', 'upload', 'formal-cmi-pref-20261012', *missing[start:start + 10]], check=True)
    published = inventory()
    for name, binding in wanted.items():
        assert published[name]['digest'] == 'sha256:' + binding['sha256'], name
        assert published[name]['size'] == binding['bytes'], name
    print(f'{len(wanted)} release assets verified against GitHub SHA256 and byte count')


if __name__ == '__main__':
    main()
