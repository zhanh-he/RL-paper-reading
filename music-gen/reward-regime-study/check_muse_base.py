"""Fail closed unless the starting weights match the published non-GRPO Muse."""
import argparse
import hashlib
import json
from pathlib import Path

PUBLISHED_SHA = 'c810b2a748579e06e960019f89a9cb98701e4ff545e2c760e8ceaa3343ff1b52'
REVISION = '0fb6b5b0a33ec48d135f046ea09524e52e502a42'


def check(model):
    model = Path(model).resolve()
    if list(model.glob('adapter*')):
        raise ValueError('Adapter files are not a pristine base checkpoint')
    digest = hashlib.sha256()
    with (model/'model.safetensors').open('rb') as source:
        for block in iter(lambda: source.read(8 << 20), b''):
            digest.update(block)
    if digest.hexdigest() != PUBLISHED_SHA:
        raise ValueError('Muse weights differ from the pinned original release')
    config = json.loads((model/'config.json').read_text())
    if config.get('peft_type') or config.get('base_model_name_or_path'):
        raise ValueError('Adapter configuration found instead of a base')
    return dict(model='bolshyC/Muse-0.6b', revision=REVISION,
        model_sha256=PUBLISHED_SHA, architectures=config['architectures'],
        config_sha256=hashlib.sha256((model/'config.json').read_bytes()).hexdigest(),
        starting_from_grpo=False, resume_from_checkpoint=False,
        source=f'https://huggingface.co/bolshyC/Muse-0.6b/tree/{REVISION}')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=check(a.model)
    if a.output.exists() and json.loads(a.output.read_text()) != result:
        raise ValueError('Base identity receipt changed')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
