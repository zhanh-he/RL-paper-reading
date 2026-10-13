"""Keep verified evaluation artifacts without exhausting scratch inode quotas."""

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

from generate_objective import sha, write


def verified_archive(run, chunk):
    receipt = run / 'archives' / (chunk.stem + '.json')
    if not receipt.exists():
        return False
    saved = json.loads(receipt.read_text())
    archive = receipt.with_suffix('.tar')
    if sha(archive) != saved['archive_sha256'] or sha(chunk) != saved['chunk_sha256']:
        raise ValueError('Archived evaluation identity changed')
    return True


def pack(run, chunk):
    if verified_archive(run, chunk):
        return
    data = json.loads(chunk.read_text())
    protocol = json.loads((run / 'protocol.json').read_text()) if (run / 'protocol.json').exists() else None
    dataset = data['rows'][0]['id'].split('_')[0]
    indices = sorted({r['prompt_index'] for r in data['rows']})
    ids = [f'{dataset}_{i:03d}_r{j}_s{([831001, 831019] if dataset == "wsb" else [931001, 931019])[j]+i}'
           for i in indices for j in range(2)]
    for row in data['rows']:
        if not {'songbench', 'mulan', 'allmusiccaps'} <= set(row['metrics']) or 'supplementary' not in row:
            raise ValueError('Do not archive unscored audio')
        if dataset == 'wsb' and ('q3o' not in row['metrics'] or row['metrics']['q3o']['status'] != 'complete'):
            raise ValueError('Q3O must complete before removing the selected audio copy')
    exports = run / 'generation-receipts.json'
    generation = {r['id']: r for r in json.loads(exports.read_text())} if exports.exists() else {}
    directories = []
    for ident in ids:
        folder = run / 'songs' / ident
        if folder.exists():
            receipt = json.loads((folder / 'evaluation_binding.json').read_text())
            if ident in generation and generation[ident] != receipt:
                raise ValueError('Generation receipt drift before archive')
            generation[ident] = receipt
        if ident not in generation:
            raise ValueError('Both original candidates must be preserved in generation metadata')
        directories.extend(p for p in [folder, run / 'evaluation/audio' / ident,
                            run / 'evaluation/receipts' / ident] if p.exists())
    paths = [chunk, *([run / 'protocol.json'] if protocol else [])]
    paths.extend(f for directory in directories for f in directory.rglob('*') if f.is_file())
    if any(f.is_symlink() for f in paths):
        raise ValueError('Only regular generated files can be archived')
    inventory = {str(f.relative_to(run)): sha(f) for f in paths}
    destination = run / 'archives' / (chunk.stem + '.tar')
    destination.parent.mkdir(exist_ok=True)
    temporary = destination.with_suffix('.part')
    with tarfile.open(temporary, 'w') as archive:
        for f in paths:
            archive.add(f, arcname=str(f.relative_to(run)), recursive=False)
    with tarfile.open(temporary, 'r') as archive:
        for member in archive:
            h = hashlib.sha256()
            with archive.extractfile(member) as source:
                while block := source.read(8 << 20):
                    h.update(block)
            if h.hexdigest() != inventory[member.name]:
                raise ValueError('Archive member hash mismatch')
    temporary.replace(destination)
    write(exports, sorted(generation.values(), key=lambda r: r['id']))
    write(destination.with_suffix('.json'), {'archive_sha256': sha(destination),
        'chunk_sha256': sha(chunk), 'files': inventory, 'dataset': dataset,
        'condition_indices': indices, 'scope': 'private full artifacts; public repo receives aggregates and MP3 only'})
    # Delete only the newly archived copies; incomplete chunks remain untouched.
    for directory in directories:
        shutil.rmtree(directory)
    cache = run / 'evaluation/.cache'
    if cache.exists():
        shutil.rmtree(cache)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--chunk', type=Path, required=True)
    a = p.parse_args()
    pack(a.run, a.chunk)


if __name__ == '__main__':
    main()
