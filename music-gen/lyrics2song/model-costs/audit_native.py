"""Count unique native nn.Parameters using meta tensors, without GPU allocation."""
import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import torch


def describe(model, pattern=None):
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    groups = Counter()
    for key,value in parameters.items():
        groups['.'.join(key.split('.')[:2])] += value.numel()
    selected = {key:list(value.shape) for key,value in parameters.items()
                if pattern and re.search(pattern,key) and value.ndim == 2}
    return dict(unique_parameters=sum(p.numel() for p in parameters.values()),
        buffers=sum(p.numel() for p in buffers.values()),groups=dict(groups),
        parameter_shapes={key:list(p.shape) for key,p in parameters.items()},
        buffer_shapes={key:list(p.shape) for key,p in buffers.items()},
        lora=dict(rank=8,modules=len(selected),parameters=sum(8*sum(s) for s in selected.values()),targets=selected))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family',choices=['hf','heart','songbloom'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    root=Path('/storage/zhanh_storage/open-music-rl-20261014')
    assets=root/'assets'
    result={}
    if a.family=='hf':
        from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
        paths={
            'yue2-core':('/home/mengh/research/YuE2-posttrain/models/YuE2-3B',AutoModelForCausalLM),
            'muse-core':('/home/mengh/research/vocal2accomp-muse/models/Muse-0.6b',AutoModelForCausalLM),
            'acestep-base':('/home/mengh/research/ACE-Step-1.5-posttrain/checkpoints/acestep-v15-base',AutoModel),
        }
        for name,(path,kind) in paths.items():
            try:
                cfg=AutoConfig.from_pretrained(path,trust_remote_code=True,local_files_only=True)
                from accelerate import init_empty_weights
                with init_empty_weights(include_buffers=False):
                    model=kind.from_config(cfg,trust_remote_code=True)
                model.tie_weights()
                pattern=(r'^decoder\.layers\.\d+\.(self_attn|cross_attn|mlp)\..*\.weight$' if name=='acestep-base'
                         else r'^model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))\.weight$')
                result[name]=describe(model,pattern)
                print(name,result[name]['unique_parameters'],result[name]['lora']['parameters'],flush=True)
                del model
            except Exception as error:
                result[name]=dict(error=f'{type(error).__name__}: {error}')
                print(name,result[name],flush=True)
    elif a.family=='heart':
        sys.path.insert(0,str(root/'sources/heartlib/src'))
        from heartlib.heartmula.modeling_heartmula import HeartMuLa
        from heartlib.heartmula.configuration_heartmula import HeartMuLaConfig
        from heartlib.heartcodec.modeling_heartcodec import HeartCodec
        from heartlib.heartcodec.configuration_heartcodec import HeartCodecConfig
        cfg=HeartMuLaConfig.from_pretrained(assets/'heartmula/HeartMuLa-oss-3B')
        with torch.device('meta'):
            model=HeartMuLa(cfg)
        result['heartmula-core']=describe(model,r'^(backbone|decoder)\.layers\.\d+\.(attn|mlp)\..*\.weight$')
        del model
        cfg=HeartCodecConfig.from_pretrained(assets/'heartmula/HeartCodec-oss')
        with torch.device('meta'):
            model=HeartCodec(cfg)
        result['heartmula-codec']=describe(model)
    else:
        sys.path.insert(0,str(root/'sources/songbloom'))
        spec=importlib.util.spec_from_file_location('native_infer',root/'sources/songbloom/infer.py')
        native=importlib.util.module_from_spec(spec); spec.loader.exec_module(native)
        from SongBloom.models.songbloom.songbloom_pl import SongBloom_PL
        cfg=native.load_config(str(assets/'songbloom/songbloom_full_150s.yaml'),parent_dir=str(assets/'songbloom'))
        original=torch.load
        def load(*args,**kwargs):
            kwargs.update(map_location='meta',mmap=True)
            return original(*args,**kwargs)
        torch.load=load
        with torch.device('meta'):
            model=SongBloom_PL(cfg)
        result['songbloom-pipeline']=describe(model,r'^model\.nar_dit\..*\.weight$')
        result['songbloom-core']=describe(model.model,r'^nar_dit\..*\.weight$')
        result['songbloom-vae']=describe(model.vae)
    a.output.write_text(json.dumps(dict(cuda_used=False,family=a.family,components=result),indent=2)+'\n')
    print({key:{k:v for k,v in value.items() if k in ['unique_parameters','buffers','error']} for key,value in result.items()},flush=True)


if __name__=='__main__':
    main()
