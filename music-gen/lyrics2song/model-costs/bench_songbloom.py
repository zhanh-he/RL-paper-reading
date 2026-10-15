"""Time one native 60-second SongBloom call; no adapters, scores, or updates."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    sys.path[:0]=[str(a.root/'sources/songbloom'),str(a.root/'code')]
    import soundfile as sf
    import torchaudio
    from load_songbloom import load
    source=a.root/'sources/songbloom/infer.py'
    spec=importlib.util.spec_from_file_location('native_infer',source)
    native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    assets=a.root/'assets/songbloom'
    cfg=native.load_config(str(assets/'songbloom_full_150s.yaml'),parent_dir=str(assets))
    cfg.max_dur=60
    start=time.perf_counter()
    sampler=load(cfg,dtype=torch.bfloat16,device=torch.device('cuda'))
    torch.cuda.synchronize();load_seconds=time.perf_counter()-start
    row=json.loads((a.root/'data.json').read_text())['valid'][0]
    refs=json.loads((a.root/'references.json').read_text())
    if isinstance(refs,dict) and 'rows' in refs:refs=refs['rows']
    ref=refs[row['condition_id']] if isinstance(refs,dict) else next(r for r in refs if r['condition_id']==row['condition_id'])
    path=Path(ref if isinstance(ref,str) else ref.get('path',ref.get('local_path',ref.get('reference_file',''))))
    if not path.is_file():raise ValueError(f'Missing native audio condition: {ref}')
    torch.manual_seed(5101);torch.cuda.manual_seed_all(5101)
    torch.cuda.synchronize();start=time.perf_counter()
    audio,sr=sf.read(path,always_2d=True,dtype='float32')
    prompt=torch.tensor(audio.mean(1))[None]
    if sr!=48000:prompt=torchaudio.functional.resample(prompt,sr,48000)
    prompt=prompt[:,:480000].to('cuda',dtype=torch.bfloat16)
    with torch.inference_mode():
        output=sampler.generate(native.clean_lyrics(row['lyrics']),prompt)
    torch.cuda.synchronize();wall=time.perf_counter()-start
    seconds=output.shape[-1]/sampler.sample_rate
    if seconds<59 or seconds>61:raise ValueError(f'Not a native60s output: {seconds}')
    receipt=dict(model='SongBloom full-150s',device=torch.cuda.get_device_name(),
        torch=torch.__version__,backend='Native BF16 /36 Euler steps /CFG1.5; first call, no kernel warmup',
        model_load_seconds=load_seconds,rows=[dict(target_seconds=60,actual_audio_seconds=seconds,
            wall_seconds=wall,rtf=wall/seconds,warmup=False,seed=5101,condition_id=row['condition_id'])],
        scope='One CMI valid input, native10s audio condition + lyrics; includes condition read/resample/encode, AR/NAR and VAE, no file save. Model initialized but first call not warmed. No scorer or update.',
        duration_control='Native cfg.max_dur=60; full checkpoint has150s limit; no300s claim')
    a.output.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
