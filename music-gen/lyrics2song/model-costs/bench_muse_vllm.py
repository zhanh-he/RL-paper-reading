"""Finite batch-one Muse vLLM benchmark with native audio-code stopping."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['model','valid','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--codec',required=True)
    a=p.parse_args()
    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    from vllm.sampling_params import RequestOutputKind
    row=json.loads(a.valid.read_text().splitlines()[0])
    tok=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True,local_files_only=True)
    prompt=tok.apply_chat_template([row['messages'][0]],tokenize=True,add_generation_prompt=True)
    low,high=[tok.convert_tokens_to_ids(f'<AUDIO_{n}>') for n in [0,16383]]
    if high-low!=16383:raise ValueError('Changed audio vocabulary')
    start=time.perf_counter()
    engine=LLM(model=str(a.model),trust_remote_code=True,dtype='bfloat16',max_model_len=12288,
        max_num_seqs=1,gpu_memory_utilization=.25,enforce_eager=False,disable_log_stats=True)
    load_seconds=time.perf_counter()-start
    results=[]
    for seconds in [4,60,300]:
        target=seconds*25;request_id=f'cost-{seconds}'
        params=SamplingParams(seed=5101,temperature=1.,top_p=1.,top_k=-1,repetition_penalty=1.,
            min_tokens=target+1024,max_tokens=target+1024,ignore_eos=True,
            output_kind=RequestOutputKind.CUMULATIVE)
        torch.cuda.synchronize();start=time.perf_counter()
        engine.llm_engine.add_request(request_id,{'prompt_token_ids':prompt},params)
        generated=[];codes=[];reached=False
        while engine.llm_engine.has_unfinished_requests():
            for output in engine.llm_engine.step():
                if output.request_id!=request_id:raise ValueError('Unexpected request')
                generated=list(output.outputs[0].token_ids)
                codes=[n-low for n in generated if low<=n<=high]
                if len(codes)>=target:
                    if len(codes)!=target:raise ValueError('Engine skipped the exact native code budget')
                    engine.llm_engine.abort_request(request_id)
                    reached=True;break
            if reached:break
        if not reached:raise ValueError(f'Audio budget not reached: {len(codes)}/{target}')
        lm_seconds=time.perf_counter()-start
        decode_start=time.perf_counter()
        req=urllib.request.Request(a.codec.rstrip('/')+'/v1/audio/decode',
            data=json.dumps(dict(tokens=codes,duration=40.96,num_steps=20,guidance_scale=1.5)).encode(),
            headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=900) as response:decoded=json.load(response)
        wall=time.perf_counter()-start
        result=dict(target_seconds=seconds,actual_audio_seconds=decoded['duration'],wall_seconds=wall,
            rtf=wall/decoded['duration'],lm_seconds=lm_seconds,
            decode_api_seconds=time.perf_counter()-decode_start,decoder_inference_seconds=decoded['inference_time'],
            warmup=seconds==4,audio_tokens=len(codes),all_generated_tokens=len(generated),seed=5101,
            generated_tokens_sha256=hashlib.sha256(json.dumps(generated).encode()).hexdigest())
        results.append(result)
        a.output.write_text(json.dumps(dict(model='Muse-0.6B',device=torch.cuda.get_device_name(),
            torch=torch.__version__,vllm=vllm.__version__,backend='vLLM BF16, batch1, graphs enabled + native MuCodec20steps',
            model_load_seconds=load_seconds,rows=results,
            scope='Fixed same CMI valid0/prompt/seed/temperature/top-p/codec settings as HF audit; exact native audio-code target with cumulative engine output and request abort; warm end-to-end including decoder HTTP/base64. Initialization excluded. Prior HF run was another allocation on same A100-SXM4-80GB type; not interleaved repeated trials. Backend RNG/kernel differences do not imply identical audio or quality.',
            duration_control='25Hz audio-code stopping, EOS ignored to budget, no grammar masking/trimming/looping'),indent=2)+'\n')
        print(json.dumps(result),flush=True)


if __name__=='__main__':main()
