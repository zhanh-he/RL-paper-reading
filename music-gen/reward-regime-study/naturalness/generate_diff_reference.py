"""Finite real native DiffRhythm2 inference; no RL, negative branch edits or fake stereo.

Input: {"cases":[{"id":...,"family":...,"arm":...,"seed":...,"style":...,"lyrics":...,
                  "cfg_strength":2.0}]}. Outputs private float32 mono WAVs and JSON receipts.
All models/assets must already exist. Shared environments/source files are not modified.
"""
import argparse, contextlib, hashlib, importlib.util, json, os, random, re, sys, time
from pathlib import Path

def file_sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def tensor_sha(tensor):return hashlib.sha256(tensor.detach().contiguous().view(__import__('torch').uint8).cpu().numpy().tobytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['source','assets','runtime-root','batch','output']:p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--seconds',type=int,default=60);p.add_argument('--steps',type=int,default=32);p.add_argument('--threads',type=int,default=4)
 a=p.parse_args();a.source=a.source.resolve();a.assets=a.assets.resolve();a.runtime_root=a.runtime_root.resolve();a.batch=a.batch.resolve();a.output=a.output.resolve()
 if a.seconds<4 or a.seconds%2 or a.steps<2:raise ValueError('Use even native seconds>=4 and steps>=2')
 batch=json.loads(a.batch.read_text());cases=batch['cases'];assert cases and len({r['id'] for r in cases})==len(cases)
 for r in cases:
  if not re.fullmatch(r'[A-Za-z0-9_-]+',r['id']):raise ValueError('Unsafe case ID')
  if not r.get('lyrics') or not r.get('style'):raise ValueError('Need explicit original lyrics and style')
  if (a.output/(r['id']+'.wav')).exists():raise FileExistsError('Refusing overwrite '+r['id'])
  if float(r.get('cfg_strength',2.))<0:raise ValueError('Nonnegative native CFG strength required')
 # Native loader expects cached MuQ files in cwd/ckpt, and assets for four model files.
 os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
 sys.dont_write_bytecode=True;sys.path.insert(0,str(a.source));os.chdir(a.runtime_root)
 import torch,numpy as np,soundfile as sf,espeakng_loader,torchaudio
 torch.set_num_threads(a.threads)
 os.environ['PHONEMIZER_ESPEAK_LIBRARY']=str(espeakng_loader.get_library_path());os.environ['ESPEAK_DATA_PATH']=str(espeakng_loader.get_data_path())
 if not torch.cuda.is_available():raise RuntimeError('CUDA required for this recorded native protocol')
 torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 rec=json.loads((a.assets/'asset-receipt.json').read_text());expected={r['file']:r['sha256'] for r in rec['files']}
 required=['model.safetensors','config.json','decoder.bin','decoder.json']
 actual={name:file_sha(a.assets/name) for name in required}
 if any(actual[name]!=expected[name] for name in required):raise ValueError('Author asset hash mismatch')
 spec=importlib.util.spec_from_file_location('diff_native_inference',a.source/'inference.py');native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
 def local_download(repo_id,filename,**kwargs):
  if repo_id!='ASLP-lab/DiffRhythm2' or filename not in required:raise ValueError('Unexpected asset request')
  return str(a.assets/filename)
 native.hf_hub_download=local_download
 a.output.mkdir(parents=True,exist_ok=True);started=time.perf_counter();model,mulan,tokenizer,decoder=native.prepare_model('ASLP-lab/DiffRhythm2','cuda');native.lrc_tokenizer=tokenizer
 model.requires_grad_(False).eval().half();mulan.requires_grad_(False).eval();decoder.requires_grad_(False).eval().half();torch.cuda.synchronize();load_seconds=time.perf_counter()-started
 text_cache={};style_cache={};conditioning={};rows=[];native_randn=torch.randn;noise_by_seed={}
 profile=dict(model='DiffRhythm2 author release',assets=actual,asset_revision=rec['files'][0]['revision'],native_source_sha256={name:file_sha(a.source/name) for name in ['inference.py','diffrhythm2/cfm.py']},device=torch.cuda.get_device_name(),torch=torch.__version__,seconds=a.seconds,steps=a.steps,load_seconds=load_seconds,batch_sha256=file_sha(a.batch),condition_interface='Actual10sSunoaudio reference encoded byMuQ-MuLan; noaudio continuationprefix iscopied',inference='Native cached block ODE; FP16 model+decoder; no adapters; no fake stereo; no waveform normalization',cfg_convention='native velocity=conditional+strength*(conditional-null); standard CFG scale=1+strength; null lyrics and style both zero',seed_protocol='External reset of Python/NumPy/Torch/CUDA per case; native seed argument is unused. All sampled initial latent noise tensors are hashed and compared for equal seeds.',reproducibility='Equal native latent noise hashes required within this run; no cross-hardware bitwise guarantee')
 for case in cases:
  case_start=time.perf_counter();style=case['style'];lyrics=case['lyrics']
  if lyrics not in text_cache:text_cache[lyrics]=torch.tensor(sum(native.parse_lyrics(lyrics),[]),device='cuda',dtype=torch.long)[None]
  with torch.inference_mode():
   if style not in style_cache:
    if case.get('reference_path'):
     if file_sha(case['reference_path'])!=case['reference_sha256']:raise ValueError('Referenceaudiohashchanged')
     ref,sr=sf.read(case['reference_path'],always_2d=True,dtype='float32');prompt=torch.from_numpy(ref.mean(1)).to('cuda')[None]
     if sr!=24000:prompt=torchaudio.functional.resample(prompt,sr,24000)
     embedding=mulan(wavs=prompt[:,:240000]).detach()
    else:embedding=mulan(texts=[style]).detach()
    style_cache[style]=embedding.half();conditioning[style]=embedding.float().cpu().reshape(-1).tolist()
    (a.output/'conditioning.json').write_text(json.dumps(dict(model='MuQ-MuLan-large',embeddings=conditioning),ensure_ascii=False,allow_nan=False)+'\n')
  seed=int(case['seed']);random.seed(seed);np.random.seed(seed%(2**32));torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);noise=[]
  def tracked_randn(*args,**kwargs):
   value=native_randn(*args,**kwargs)
   if tuple(value.shape)==(1,model.block_size,model.num_channels):noise.append(tensor_sha(value))
   return value
  torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();sample_start=time.perf_counter()
  try:
   torch.randn=tracked_randn
   with torch.inference_mode():latent=model.sample_block_cache(text_cache[lyrics],a.seconds*5,style_cache[style],steps=a.steps,cfg_strength=float(case.get('cfg_strength',2.)),process_bar=False)
  finally:torch.randn=native_randn
  torch.cuda.synchronize();ode_seconds=time.perf_counter()-sample_start
  if len(noise)!=(a.seconds*5+model.block_size-1)//model.block_size:raise ValueError('Did not capture exactly one initial latent noise per native block')
  if seed in noise_by_seed and noise_by_seed[seed]!=noise:raise ValueError('Same-seed initial noise mismatch; paired claim invalid')
  noise_by_seed[seed]=noise
  decode_start=time.perf_counter()
  with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
   torch.manual_seed(42);torch.cuda.manual_seed_all(42)
   with torch.inference_mode():wave=decoder.decode_audio(latent.transpose(1,2),overlap=5,chunk_size=20)
  torch.cuda.synchronize();decode_seconds=time.perf_counter()-decode_start
  array=wave.float().cpu().numpy().reshape(-1);sr=int(decoder.h.sampling_rate)
  if sr!=48000 or not np.isfinite(array).all() or abs(len(array)/sr-a.seconds)>.001:raise ValueError('Invalid native output')
  path=a.output/(case['id']+'.wav');sf.write(path,array,sr,subtype='FLOAT')
  record=dict(case,source_kind='actual_generation',path=str(path),audio_sha256=file_sha(path),sample_rate=sr,channels=1,duration_s=len(array)/sr,latent_sha256=tensor_sha(latent),initial_noise_sha256=noise,standard_cfg_scale=1+float(case.get('cfg_strength',2.)),ode_seconds=ode_seconds,decode_seconds=decode_seconds,decoder_seed=42,end_to_end_seconds=time.perf_counter()-case_start,gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),rms_dbfs=float(20*np.log10(max(np.sqrt(np.mean(array.astype(float)**2)),1e-12))),peak=float(np.max(np.abs(array))),human_rating=None)
  rows.append(record);(a.output/'run.json').write_text(json.dumps(dict(status='running',profile=profile,samples=rows),ensure_ascii=False,indent=2,allow_nan=False)+'\n');print(json.dumps({k:record[k] for k in ['id','duration_s','audio_sha256','end_to_end_seconds','gpu_peak_allocated_bytes']}) ,flush=True)
 (a.output/'run.json').write_text(json.dumps(dict(status='complete',profile=profile,samples=rows),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
