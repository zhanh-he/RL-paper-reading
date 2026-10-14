"""Execute a command inside the existing hash-verified Gadi OS image."""
import hashlib
import json
import os
from pathlib import Path
import sys

config = json.loads(Path(sys.argv[1]).read_text())
digest = hashlib.sha256()
with Path(config['image']).open('rb') as stream:
    while block := stream.read(8 << 20): digest.update(block)
if digest.hexdigest() != config['image_sha256']: raise ValueError('OS image hash changed')
command = [config['binary'], 'exec', '--nv', '--bind', ','.join(config['binds']), config['image'], *sys.argv[2:]]
os.execv(command[0], command)
