"""Standard-library-only entry into the existing hash-bound evaluator image."""
import hashlib
import json
import os
from pathlib import Path
import sys


def enter(work):
    runtime = Path(work)/'container-runtime.json'
    if not runtime.exists():
        return
    config = json.loads(runtime.read_text())
    if os.environ.get('EVAL_CONTAINER_ACTIVE') == config['image_sha256']:
        return
    digest = hashlib.sha256()
    with Path(config['image']).open('rb') as source:
        for block in iter(lambda: source.read(8 << 20), b''):
            digest.update(block)
    if digest.hexdigest() != config['image_sha256']:
        raise ValueError('Evaluation OS image changed after preparation')
    command = [config['binary'], 'exec', '--nv', '--bind', ','.join(config['binds']),
               '--env', 'EVAL_CONTAINER_ACTIVE='+config['image_sha256'],
               config['image'], sys.executable, *sys.argv]
    os.execv(command[0], command)
