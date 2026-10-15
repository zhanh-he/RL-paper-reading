"""Read weight shapes on CPU only; never allocate CUDA or read tensor payloads."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import struct


def safe_header(path):
    with path.open('rb') as source:
        size = struct.unpack('<Q', source.read(8))[0]
        if size > 64 << 20:
            raise ValueError('Unexpected safetensors header size')
        raw = source.read(size)
    header = json.loads(raw)
    return {key:value['shape'] for key,value in header.items() if key != '__metadata__'}, hashlib.sha256(raw).hexdigest()


def torch_shapes(path):
    import torch
    state = torch.load(path, map_location='meta', mmap=True, weights_only=False)
    for key in ['state_dict','model','generator']:
        if isinstance(state,dict) and key in state and isinstance(state[key],dict):
            state = state[key]
            break
    return {key:list(value.shape) for key,value in state.items() if isinstance(value,torch.Tensor)}


def count_lora(shapes, pattern, rank):
    selected = {key:shape for key,shape in shapes.items() if re.search(pattern,key) and len(shape) == 2}
    return dict(rank=rank,modules=len(selected),parameters=sum(rank*sum(shape) for shape in selected.values()),
                targets=selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    yue = Path('/home/mengh/research/YuE2-posttrain/models')
    native = Path('/storage/zhanh_storage/open-music-rl-20261014/assets')
    ace = Path('/home/mengh/research/ACE-Step-1.5-posttrain/checkpoints')
    muse = Path('/home/mengh/research/vocal2accomp-muse')
    components = {
        'yue2-core': [yue/'YuE2-3B/model.safetensors'],
        'yue2-vae': [yue/'YuE2-Vae/model.safetensors'],
        'muse-core': [muse/'models/Muse-0.6b/model.safetensors'],
        'diffrhythm2-core': [native/'diffrhythm2/model.safetensors'],
        'diffrhythm2-decoder': [native/'diffrhythm2/decoder.bin'],
        'heartmula-core': sorted((native/'heartmula/HeartMuLa-oss-3B').glob('model-*.safetensors')),
        'heartmula-codec': sorted((native/'heartmula/HeartCodec-oss').glob('model-*.safetensors')),
        'songbloom-core': [native/'songbloom/songbloom_full_150s.pt'],
        'songbloom-vae': [native/'songbloom/autoencoder_music_dsp1920.ckpt'],
        'acestep-base': [ace/'acestep-v15-base/model.safetensors'],
        'acestep-turbo': [ace/'acestep-v15-turbo/model.safetensors'],
        'acestep-lm': [ace/'acestep-5Hz-lm-1.7B/model.safetensors'],
        'acestep-text': [ace/'Qwen3-Embedding-0.6B/model.safetensors'],
        'acestep-vae': [ace/'vae/diffusion_pytorch_model.safetensors'],
    }
    patterns = {
        'yue2-core': (r'\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))\.weight$',8),
        'muse-core': (r'\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))\.weight$',8),
        'diffrhythm2-core': (r'transformer_blocks\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))\.weight$',8),
        'heartmula-core': (r'^(backbone|decoder)\.layers\.\d+\.(attn|mlp)\..*\.weight$',8),
        'songbloom-core': (r'^diffusion\.nar_dit\..*\.weight$',8),
        'acestep-base': (r'^decoder\.layers\.\d+\.(self_attn|cross_attn|mlp)\..*\.weight$',8),
    }
    result = {}
    for name, files in components.items():
        shapes, receipts = {}, []
        try:
            if not files:
                raise FileNotFoundError(name)
            for path in files:
                if path.suffix == '.safetensors':
                    values,digest = safe_header(path)
                else:
                    values,digest = torch_shapes(path),None
                if shapes.keys() & values.keys():
                    raise ValueError('Duplicate shard keys')
                shapes.update(values)
                receipts.append(dict(path=str(path),bytes=path.stat().st_size,header_sha256=digest))
            groups = Counter()
            for key,shape in shapes.items():
                groups['.'.join(key.split('.')[:2])] += math.prod(shape)
            result[name] = dict(checkpoint_elements=sum(math.prod(s) for s in shapes.values()),
                tensor_count=len(shapes),files=receipts,groups=dict(groups),shapes=shapes,
                note='State tensors; buffers and tied aliases require native-model reconciliation before publishing as parameters')
            if name in patterns:
                result[name]['lora_shape_plan'] = count_lora(shapes,*patterns[name])
            print(name, result[name]['checkpoint_elements'],result[name].get('lora_shape_plan',{}).get('parameters'),flush=True)
        except Exception as error:
            result[name] = dict(error=f'{type(error).__name__}: {error}')
            print(name,result[name]['error'],flush=True)
    args.output.write_text(json.dumps(dict(snapshot_at=datetime.now(timezone.utc).isoformat(),cuda_used=False,components=result),indent=2)+'\n')


if __name__ == '__main__':
    main()
