"""Use exactly the same frozen baseline audio across YuE2 reward arms."""
import argparse
import shutil
import sys
from pathlib import Path
import train_yue2 as core
from rewards import sha

p = argparse.ArgumentParser(add_help=False)
p.add_argument('--baseline-root', type=Path, required=True)
a, rest = p.parse_known_args()
original = core.evaluate


def evaluate(pipe, args, data, step):
    if step == 0:
        import json
        receipt = json.loads((a.baseline_root/'receipt.json').read_text())
        if receipt['n'] != 30 or receipt['step'] != 0:
            raise ValueError('Baseline is not a complete frozen validation cohort')
        for i, row in enumerate(receipt['rows']):
            source = a.baseline_root/f'{i:03d}'
            if sha(source/'audio.flac') != row['audio_sha256']:
                raise ValueError('Frozen baseline changed')
            destination = args.output/'valid_000000'/f'{i:03d}'
            destination.mkdir(parents=True, exist_ok=True)
            for name in ['audio.flac', 'rollout.json']:
                if (destination/name).exists():
                    if sha(destination/name) != sha(source/name): raise ValueError('Baseline pairing changed')
                else: shutil.copy2(source/name, destination/name)
        core.write(args.output/'baseline-identity.json', {'source_receipt_sha256': sha(a.baseline_root/'receipt.json'),
            'wrapper_sha256': sha(Path(__file__)), 'scope': 'identical audio/seeds across all three reward arms'})
    return original(pipe,args,data,step)


core.evaluate = evaluate
sys.argv = [sys.argv[0], *rest]
core.main()
