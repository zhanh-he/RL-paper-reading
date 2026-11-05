"""Finite native SongBloom generations from actual audio reference+lyrics.

No text-style/negative-prompt input is invented. Checkpoint provenance remains
community-restored (existing asset receipt), not independently author-matched.
Input {cases:[{id,seed,lyrics,reference_path,reference_sha256,cfg_coef:1.5}]}.
"""
import argparse,hashlib,importlib.util,json,os,random,re,sys,time
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['source','assets','runtime-root','loader','batch','output']:p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--seconds',type=int,default=60);p.add_argument('--steps',type=int,default=36);p.add_argument('--threads',type=int,default=4);a=p.parse_args()
 for name in ['source','assets','runtime_root','loader','batch','output']:setattr(a,name,getattr(a,name).resolve())
 if not 10<=a.seconds<=150 or a.steps<2:raise ValueError('Native full150s bounds:10–150s,>=2steps')
 cases=json.loads(a.batch.read_text())['cases'];assert cases and len({r['id'] for r in cases})==len(cases)
 for r in cases:
  if not re.fullmatch(r'[A-Za-z0-9_-]+',r['id']):raise ValueError('Unsafe ID')
  if (a.output/(r['id']+'.wav')).exists():raise FileExistsError('Refusing overwrite')
  if not r.get('lyrics') or not r.get('reference_path') or not r.get('reference_sha256'):raise ValueError('Need real reference+identity+lyrics')
  if sha(r['reference_path'])!=r['reference_sha256']:raise ValueError('Reference audio changed')
 os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',DISABLE_FLASH_ATTN='1');sys.dont_write_bytecode=True
 sys.path.insert(0,str(a.source));os.chdir(a.runtime_root)
 import torch,torchaudio,numpy as np,soundfile as sf
 torch.set_num_threads(a.threads);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 if not torch.cuda.is_available():raise RuntimeError('CUDA required')
 receipt=json.loads((a.assets/'asset-receipt.json').read_text());expected={r['file']:r['sha256'] for r in receipt['files']}
 names=['songbloom_full_150s.pt','autoencoder_music_dsp1920.ckpt','songbloom_full_150s.yaml','stable_audio_1920_vae.json','vocab_g2p.yaml'];actual={name:sha(a.assets/name) for name in names}
 if any(actual[name]!=expected[name] for name in names):raise ValueError('Archived asset hash mismatch')
 native=module('songbloom_native_inference',a.source/'infer.py');loader=module('songbloom_checked_loader',a.loader)
 cfg=native.load_config(str(a.assets/'songbloom_full_150s.yaml'),parent_dir=str(a.assets));cfg.max_dur=a.seconds
 start=time.perf_counter();sampler=loader.load(cfg,dtype=torch.bfloat16,device=torch.device('cuda'));sampler.set_generation_params(steps=a.steps);torch.cuda.synchronize();load_seconds=time.perf_counter()-start
 a.output.mkdir(parents=True,exist_ok=True);rows=[]
 profile=dict(model='SongBloom full150s community-restored checkpoint',asset_source_status=receipt.get('source_status'),asset_repository=receipt['files'][0]['repo'],asset_revision=receipt['files'][0]['revision'],assets=actual,loader_sha256=sha(a.loader),source_sha256={'infer.py':sha(a.source/'infer.py'),'songbloom_pl.py':sha(a.source/'SongBloom/models/songbloom/songbloom_pl.py')},checkpoint_compatibility=sampler.checkpoint_compatibility,device=torch.cuda.get_device_name(),torch=torch.__version__,load_seconds=load_seconds,native_requested_seconds=a.seconds,steps=a.steps,settings=dict(sampler.generation_params),batch_sha256=sha(a.batch),condition_interface='Actual first10s mono reference audio + cleaned lyrics; no text style prompt or negative prompt interface.',scope='Real native AR-sketch+acoustic NAR+VAE generation; no RL/update/normalization; native output may be60.16s for60s target',reproducibility='External Python/NumPy/Torch/CUDA reset per case; equal seed across models does not imply identical latent noise or music')
 for r in cases:
  start=time.perf_counter();reference,sr=sf.read(r['reference_path'],always_2d=True,dtype='float32');prompt=torch.from_numpy(reference.mean(1))[None]
  if sr!=48000:prompt=torchaudio.functional.resample(prompt,sr,48000)
  if prompt.shape[-1]<480000:raise ValueError('Need complete native10s reference')
  prompt=prompt[:,:480000].to('cuda',dtype=torch.bfloat16);clean=native.clean_lyrics(r['lyrics']);seed=int(r['seed']);random.seed(seed);np.random.seed(seed%(2**32));torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
  sampler.set_generation_params(cfg_coef=float(r.get('cfg_coef',1.5)));torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();gen_start=time.perf_counter()
  with torch.inference_mode():wave=sampler.generate(clean,prompt)
  torch.cuda.synchronize();gen_seconds=time.perf_counter()-gen_start;array=wave[0].detach().float().cpu().numpy().T;rate=int(sampler.sample_rate)
  if array.ndim==1:array=array[:,None]
  duration=len(array)/rate
  if rate!=48000 or not np.isfinite(array).all() or abs(duration-a.seconds)>1:raise ValueError('Invalid native output')
  path=a.output/(r['id']+'.wav');sf.write(path,array,rate,subtype='FLOAT')
  row=dict(r,path=str(path),source_kind='actual_generation',audio_sha256=sha(path),reference_excerpt_rule='Mean-fold to mono, resample48k then first10s; no amplitude normalization',sample_rate=rate,channels=array.shape[1],duration_s=duration,generation_seconds=gen_seconds,end_to_end_seconds=time.perf_counter()-start,gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),rms_dbfs=float(20*np.log10(max(np.sqrt(np.mean(array.astype(float)**2)),1e-12))),peak=float(np.max(np.abs(array))),human_rating=None)
  rows.append(row);(a.output/'run.json').write_text(json.dumps(dict(status='running',profile=profile,samples=rows),ensure_ascii=False,indent=2,allow_nan=False)+'\n');print(json.dumps({k:row[k] for k in ['id','duration_s','audio_sha256','end_to_end_seconds','gpu_peak_allocated_bytes']}),flush=True)
 (a.output/'run.json').write_text(json.dumps(dict(status='complete',profile=profile,samples=rows),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
