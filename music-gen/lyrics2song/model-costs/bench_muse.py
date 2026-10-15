"""Time native Muse token generation plus the official MuCodec HTTP decoder."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList


class AudioBudget(StoppingCriteria):
    def __init__(self, low, high, target):
        self.low,self.high,self.target,self.count=low,high,target,0

    def __call__(self,input_ids,scores,**kwargs):
        self.count += self.low <= int(input_ids[0,-1]) <= self.high
        return torch.tensor([self.count>=self.target],device=input_ids.device)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['model','valid','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--codec',required=True)
    a=p.parse_args()
    with a.valid.open() as source:
        row=json.loads(next(source))
    start=time.perf_counter()
    tok=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(a.model,trust_remote_code=True,local_files_only=True,
             torch_dtype=torch.bfloat16,attn_implementation='sdpa').eval().cuda()
    torch.cuda.synchronize(); load_seconds=time.perf_counter()-start
    ids=tok.apply_chat_template([row['messages'][0]],tokenize=True,add_generation_prompt=True)
    inputs=torch.tensor([ids],device='cuda')
    low,high=[tok.convert_tokens_to_ids(f'<AUDIO_{value}>') for value in [0,16383]]
    if high-low!=16383:
        raise ValueError('Changed native audio vocabulary')
    results=[]
    for seconds in [4,60,300]:
        target=int(seconds*25)
        torch.manual_seed(5101); torch.cuda.manual_seed_all(5101)
        torch.cuda.synchronize(); start=time.perf_counter()
        with torch.inference_mode():
            output=model.generate(input_ids=inputs,attention_mask=torch.ones_like(inputs),
                do_sample=True,temperature=1.,top_p=1.,top_k=0,repetition_penalty=1.,
                max_new_tokens=target+1024,min_new_tokens=target+1024,pad_token_id=tok.eos_token_id,
                stopping_criteria=StoppingCriteriaList([AudioBudget(low,high,target)]))
        torch.cuda.synchronize(); lm_seconds=time.perf_counter()-start
        generated=output[0,len(ids):].tolist()
        codes=[value-low for value in generated if low<=value<=high]
        if len(codes)!=target:
            raise ValueError(f'Expected {target} native audio codes, got {len(codes)}')
        decode_start=time.perf_counter()
        request=urllib.request.Request(a.codec.rstrip('/')+'/v1/audio/decode',
            data=json.dumps(dict(tokens=codes,duration=40.96,num_steps=20,guidance_scale=1.5)).encode(),
            headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=900) as response:
            decoded=json.load(response)
        wall=time.perf_counter()-start
        result=dict(target_seconds=seconds,actual_audio_seconds=decoded['duration'],wall_seconds=wall,
            rtf=wall/decoded['duration'],lm_seconds=lm_seconds,decode_api_seconds=time.perf_counter()-decode_start,
            decoder_inference_seconds=decoded['inference_time'],warmup=seconds==4,
            audio_tokens=len(codes),all_generated_tokens=len(generated),seed=5101,
            generated_tokens_sha256=hashlib.sha256(json.dumps(generated).encode()).hexdigest())
        results.append(result)
        a.output.write_text(json.dumps(dict(model='Muse-0.6B',device=torch.cuda.get_device_name(),
            torch=torch.__version__,backend='Transformers SDPA BF16 + native MuCodec HTTP FP16 diffusion20steps',
            model_load_seconds=load_seconds,rows=results,
            scope='One fixed CMI valid condition; warm LM+decoder API including transfer/base64; load, queue, scorer and MP3 encode excluded',
            duration_control='Native25Hz audio codes; EOS suppressed to audio-token target; no waveform loops'),indent=2)+'\n')
        print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
