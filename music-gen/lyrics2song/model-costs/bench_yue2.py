"""Independent interactive inference timing; no training/scoring/checkpoint updates."""
import argparse
import json
from pathlib import Path
import time
import torch
from yue2 import YuE2Pipeline


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['model','vae','data','output']:
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    row=json.loads(a.data.read_text())['valid'][0]
    start=time.perf_counter()
    pipe=YuE2Pipeline.from_pretrained(a.model,vae=a.vae,device='cuda',backend='torch-eager',
                                    memory_budget_gib=64,progress=False)
    pipe._load_model()
    model_load=time.perf_counter()-start
    results=[]
    for seconds in [4,60,300]:
        tokens=int(seconds*25)
        torch.cuda.synchronize(); start=time.perf_counter()
        with torch.inference_mode():
            song=pipe(style=row['style'],lyrics=row['lyrics'],seed=5101,
                      semantic_sampling=dict(min_tokens=tokens,max_tokens=tokens+1))
        torch.cuda.synchronize(); elapsed=time.perf_counter()-start
        duration=len(song.audio)/song.sample_rate
        result=dict(target_seconds=seconds,actual_audio_seconds=duration,wall_seconds=elapsed,
            rtf=elapsed/duration,timing=song.timing,semantic_tokens=len(song.semantic.tokens),
            truncated=song.truncated,seed=5101,warmup=seconds==4)
        results.append(result)
        a.output.write_text(json.dumps(dict(model='YuE2-3B',device=torch.cuda.get_device_name(),
            torch=torch.__version__,backend='torch-eager',model_load_seconds=model_load,rows=results,
            scope='One fixed CMI valid condition; model load separate; native plan+semantic+NAR+VAE; no scoring or MP3 encoding',
            duration_control='25 semantic tokens/s; EOS suppressed to target token budget, no looping waveform'),indent=2)+'\n')
        print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
