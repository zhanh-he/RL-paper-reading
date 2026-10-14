"""Independent full 30-condition Muse validation with one shared frozen baseline."""
import argparse
import base64
import fcntl
import io
import json
import os
import urllib.request
from pathlib import Path
import numpy as np
import soundfile as sf
from rewards import score_remote, sha

VALID_SHA = 'ed5c5f36af48ef6ce81d15c7b15b429fb250ec4a52a04a43e55d96ec3cafb59d'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True); p.add_argument('--valid', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--adapter', type=Path, required=True); p.add_argument('--scorer', required=True)
    p.add_argument('--codec', required=True); p.add_argument('--reward', required=True)
    a = p.parse_args()
    if sha(a.valid) != VALID_SHA: raise ValueError('Frozen validation membership changed')
    rows = [json.loads(line) for line in a.valid.read_text().splitlines()]
    if len(rows) != 30: raise ValueError('Full validation requires all 30 conditions')
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    tokenizer = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True, local_files_only=True)
    low = tokenizer.convert_tokens_to_ids('<AUDIO_0>'); high = tokenizer.convert_tokens_to_ids('<AUDIO_16383>')
    if high-low != 16383: raise ValueError('Audio vocabulary changed')
    engine = LLM(model=str(a.model), trust_remote_code=True, dtype='bfloat16', enforce_eager=True,
                 max_model_len=4096, gpu_memory_utilization=.2, max_num_seqs=2, enable_lora=True, max_lora_rank=8)
    for step, folder, adapter in [(0,a.baseline,None),(270,a.output/'valid_000270',a.adapter)]:
        folder.mkdir(parents=True, exist_ok=True)
        with (folder/'generation.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            pending = [i for i in range(30) if not (folder/f'{i:03d}'/'generation.json').exists()]
            prompts = [tokenizer.apply_chat_template([rows[i]['messages'][0]], tokenize=False, add_generation_prompt=True) for i in pending]
            params = [SamplingParams(seed=5101+i, max_tokens=512, temperature=1.,top_p=1.,top_k=-1,repetition_penalty=1.) for i in pending]
            request = LoRARequest('regime-final',1,str(adapter)) if adapter else None
            generated = engine.generate(prompts, params, lora_request=request, use_tqdm=False) if prompts else []
            for i, result in zip(pending, generated):
                destination = folder/f'{i:03d}'; destination.mkdir(exist_ok=True)
                ids = list(result.outputs[0].token_ids); codes = [n-low for n in ids if low<=n<=high]
                meta = {'index':i,'seed':5101+i,'generated_ids':ids,'audio_tokens':codes,
                        'adapter_sha256':sha(adapter/'adapter_model.safetensors') if adapter else None,
                        'valid_sha256': VALID_SHA, 'max_tokens':512, 'forced_length':False}
                if not codes:
                    meta['invalid'] = 'no AUDIO tokens'; meta['audio_sha256'] = None
                else:
                    req = urllib.request.Request(a.codec.rstrip('/')+'/v1/audio/decode',
                        data=json.dumps({'tokens':codes,'duration':10.24,'num_steps':20}).encode(),
                        headers={'Content-Type':'application/json'})
                    with urllib.request.urlopen(req, timeout=600) as response: decoded=json.load(response)
                    audio, sr = sf.read(io.BytesIO(base64.b64decode(decoded['audio_base64'])),dtype='float32')
                    if not np.isfinite(audio).all():
                        meta['invalid']='nonfinite decoded audio';meta['audio_sha256']=None
                    else:
                        sf.write(destination/'audio.flac',audio,sr,subtype='PCM_24')
                        meta.update(audio_sha256=sha(destination/'audio.flac'),seconds=len(audio)/sr)
                (destination/'generation.json').write_text(json.dumps(meta)+'\n')
        valid_paths=[]; results=[]; indexes=[]
        for i in range(30):
            meta=json.loads((folder/f'{i:03d}'/'generation.json').read_text())
            if meta.get('invalid'):
                results.append({'index':i,'mean':0.,'invalid':meta['invalid'],'audio_sha256':None})
            else:
                path=folder/f'{i:03d}'/'audio.flac'
                if sha(path)!=meta['audio_sha256']: raise ValueError('Validation audio changed')
                indexes.append(i);valid_paths.append(path)
        for i,result in zip(indexes,score_remote(a.scorer,valid_paths,a.reward) if valid_paths else []):
            results.append({'index':i,**result})
        results.sort(key=lambda r:r['index'])
        report={'step':step,'n':30,'mean_reward':float(np.mean([r['mean'] for r in results])),
                'rows':results,'scope':'all 30 fixed validation conditions; invalid outputs retain zero objective; not test',
                'invalid_n':sum('invalid' in r or r.get('guard',{}).get('invalid',False) for r in results)}
        destination=a.output/f'valid_{step:06d}'
        destination.mkdir(parents=True,exist_ok=True)
        (destination/'receipt.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(f'validation step={step} n=30 mean={report["mean_reward"]}',flush=True)


if __name__=='__main__':main()
