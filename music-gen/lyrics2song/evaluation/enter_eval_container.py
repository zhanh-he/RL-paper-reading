"""Re-exec Gadi evaluation in its hash-bound compatible operating-system image."""

import json
import os
import sys
from pathlib import Path

from generate_objective import sha


def enter(work):
    runtime = work / 'container-runtime.json'
    if not runtime.exists():
        return
    config = json.loads(runtime.read_text())
    if os.environ.get('EVAL_CONTAINER_ACTIVE') == config['image_sha256']:
        return
    if sha(Path(config['image'])) != config['image_sha256']:
        raise ValueError('Evaluation OS image changed after preparation')
    command = [config['binary'], 'exec', '--nv', '--bind', ','.join(config['binds']),
               '--env', 'EVAL_CONTAINER_ACTIVE=' + config['image_sha256'],
               config['image'], sys.executable, *sys.argv]
    os.execv(command[0], command)
