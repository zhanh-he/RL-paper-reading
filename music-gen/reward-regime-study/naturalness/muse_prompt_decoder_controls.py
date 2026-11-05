"""Two-stage native Muse prompt + exact-token MuCodec guidance controls.

Public APIs only; no private pilot import, RL, waveform transforms or inferred
human labels. Run lm and decode with their respective existing Python envs.
"""
import argparse,copy,hashlib,json,os,random,re,sys,time
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def codes_sha(values):return hashlib.sha256(json.dumps(values,separators=(',',':')).encode()).hexdigest()
def t_sha(t):
 import torch
 return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
def setup():
 os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1');sys.dont_write_bytecode=True
 import torch
 torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 if not torch.cuda.is_available():raise RuntimeError('Recorded native protocol requires CUDA')
 return torch

def lm(a):
 torch=setup();import numpy as np
 from transformers import AutoModelForCausalLM,AutoTokenizer,StoppingCriteria,StoppingCriteriaList
 model_path=a.model.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);batch=json.loads(a.batch.read_text());cases=batch['cases']
 for r in cases:
  if not re.fullmatch(r'[A-Za-z0-9_-]+',r['id']):raise ValueError('UnsafecaseID')
  if (out/(r['id']+'.json')).exists():raise FileExistsError('Refusing token artifact overwrite')
 started=time.perf_counter();tok=AutoTokenizer.from_pretrained(model_path,trust_remote_code=True,local_files_only=True)
 model=AutoModelForCausalLM.from_pretrained(model_path,dtype=torch.bfloat16,trust_remote_code=True,local_files_only=True,attn_implementation='sdpa').to('cuda').eval();model.requires_grad_(False)
 torch.cuda.synchronize();load_seconds=time.perf_counter()-started;low=tok.convert_tokens_to_ids('<AUDIO_0>');high=tok.convert_tokens_to_ids('<AUDIO_16383>')
 if high-low!=16383:raise ValueError('Muse audio vocabulary is not contiguous')
 class NativeAudioBudget(StoppingCriteria):
  def __init__(self,prompt_len):self.prompt_len=prompt_len
  def __call__(self,input_ids,scores,**kwargs):
   suffix=input_ids[:,self.prompt_len:];return (((suffix>=low)&(suffix<=high)).sum(1)>=a.audio_tokens)
 profile=dict(model='Muse0.6B localcheckpoint',checkpoint_sha256=sha(model_path/'model.safetensors'),model_config_sha256=sha(model_path/'config.json'),torch=torch.__version__,device=torch.cuda.get_device_name(),dtype='BF16',backend='HF AutoModelForCausalLM SDPA',load_seconds=load_seconds,temperature=.9,top_p=.9,top_k=0,repetition_penalty=1.3,requested_audio_tokens=a.audio_tokens,audio_budget_protocol='EOS ignored; stop immediately at exact requested AUDIO-code count, cap at budget+1024 total new tokens. No grammar masking, repetition, token padding or waveform duration claim.',prompt_template='Muse native style sentence + structural [Section][desc:...][lyrics:...] blocks',batch_sha256=sha(a.batch))
 rows=[]
 for r in cases:
  seed=int(r['seed']);random.seed(seed);np.random.seed(seed%(2**32));torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
  prompt=r['prompt'];ids=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,add_generation_prompt=True);input_ids=torch.tensor([ids],device='cuda');cfg=copy.deepcopy(model.generation_config);cfg.eos_token_id=[]
  torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
  with torch.inference_mode():
   result=model.generate(input_ids=input_ids,attention_mask=torch.ones_like(input_ids),generation_config=cfg,do_sample=True,eos_token_id=[],temperature=.9,top_p=.9,top_k=0,repetition_penalty=1.3,max_new_tokens=a.audio_tokens+1024,stopping_criteria=StoppingCriteriaList([NativeAudioBudget(len(ids))]),pad_token_id=tok.eos_token_id)
  torch.cuda.synchronize();generated=result[0,len(ids):].tolist();audio=[v-low for v in generated if low<=v<=high]
  artifact=dict(r,prompt_ids=ids,generated_ids=generated,audio_tokens=audio,audio_tokens_sha256=codes_sha(audio),requested_audio_tokens=a.audio_tokens,audio_token_count=len(audio),audio_budget_met=len(audio)==a.audio_tokens,inference_seconds=time.perf_counter()-start,gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated())
  token_file=out/(r['id']+'.json');token_file.write_text(json.dumps(artifact,ensure_ascii=False,indent=2,allow_nan=False)+'\n');rows.append(dict(id=r['id'],family=r.get('family'),arm=r.get('arm'),token_file=str(token_file),token_file_sha256=sha(token_file),audio_tokens_sha256=artifact['audio_tokens_sha256'],audio_budget_met=artifact['audio_budget_met'],audio_token_count=len(audio),seed=seed))
  (out/'run.json').write_text(json.dumps(dict(status='running',profile=profile,samples=rows),ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:artifact[k] for k in ['id','audio_token_count','audio_budget_met','inference_seconds','gpu_peak_allocated_bytes']}),flush=True)
 (out/'run.json').write_text(json.dumps(dict(status='complete' if all(r['audio_budget_met'] for r in rows) else 'complete_with_unmet_budgets',profile=profile,samples=rows),ensure_ascii=False,indent=2)+'\n')

def decode(a):
 torch=setup();import numpy as np,soundfile as sf
 source=a.mucodec.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(source));os.chdir(source);from generate import MuCodec
 weights=source/'weights/mucodec.pt'
 if not weights.exists():weights=source/'ckpt/mucodec.pt'
 if not weights.is_file():raise FileNotFoundError('No actual MuCodec weights')
 token_run=json.loads(a.tokens.read_text());by={r['id']:r for r in token_run['samples']};variants=json.loads(a.plan.read_text())['variants']
 for v in variants:
  if not re.fullmatch(r'[A-Za-z0-9_-]+',v['id']):raise ValueError('UnsafeID')
  if (out/(v['id']+'.wav')).exists():raise FileExistsError('Refusing overwrite')
  if v['source_case_id'] not in by:raise ValueError('Unknown token source')
 start=time.perf_counter();decoder=MuCodec(str(weights),layer_num=7,load_main_model=True,device='cuda:0');torch.cuda.synchronize();load_seconds=time.perf_counter()-start
 profile=dict(model='Official MuCodec decoder for exact Muse codes',decoder_weight_sha256=sha(weights),decoder_api_source_sha256=sha(source/'generate.py'),torch=torch.__version__,device=torch.cuda.get_device_name(),load_seconds=load_seconds,native_window_duration=a.duration,num_steps=a.steps,seed=a.seed,token_run_sha256=sha(a.tokens),decode_plan_sha256=sha(a.plan),scope='Exact AUDIO codes held fixed across base guidance variants; native code2sound; no normalization, filter, fake stereo or RL. Semantic pair uses genuinely different Muse codes and is labeled separately.')
 rows=[];noise_by_codes={};native_randn=torch.randn;native_randn_like=torch.randn_like
 for v in variants:
  r=by[v['source_case_id']];path=Path(r['token_file']);assert sha(path)==r['token_file_sha256'];artifact=json.loads(path.read_text());audio=artifact['audio_tokens']
  if len(audio)<32 or codes_sha(audio)!=artifact['audio_tokens_sha256']:raise ValueError('Invalid/unbound native audio codes')
  codes=torch.tensor(audio,dtype=torch.long).view(1,1,-1);noise=[]
  def capture(fn,kind,*args,**kwargs):
   value=fn(*args,**kwargs);noise.append(dict(kind=kind,shape=list(value.shape),dtype=str(value.dtype),sha256=t_sha(value)));return value
  def tracked_randn(*args,**kwargs):return capture(native_randn,'randn',*args,**kwargs)
  def tracked_randn_like(*args,**kwargs):return capture(native_randn_like,'randn_like',*args,**kwargs)
  random.seed(a.seed);np.random.seed(a.seed%(2**32));torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter();native_inference_codes=decoder.model.inference_codes;used_guidance=[]
  def traced_inference_codes(*args,**kwargs):
   if v.get('override_guidance',False):kwargs['guidance_scale']=float(v['guidance_scale'])
   used_guidance.append(float(kwargs.get('guidance_scale',1.5)));return native_inference_codes(*args,**kwargs)
  try:
   decoder.model.inference_codes=traced_inference_codes
   torch.randn=tracked_randn;torch.randn_like=tracked_randn_like
   with torch.inference_mode():wave=decoder.code2sound(codes,prompt=None,duration=a.duration,guidance_scale=float(v['guidance_scale']),num_steps=a.steps,disable_progress=True)
  finally:torch.randn=native_randn;torch.randn_like=native_randn_like;decoder.model.inference_codes=native_inference_codes
  torch.cuda.synchronize();seconds=time.perf_counter()-start;array=wave.detach().float().cpu().numpy()
  if array.ndim==3:
   if array.shape[0]!=1:raise ValueError('Unexpected batch')
   array=array[0]
  if array.ndim==1:array=array[None,:]
  if array.ndim!=2 or not np.isfinite(array).all():raise ValueError('Invalid native waveform')
  array=array.T;key=artifact['audio_tokens_sha256'];prior=noise_by_codes.get(key);noise_equal=(prior==noise) if prior is not None else None;noise_by_codes.setdefault(key,noise)
  if noise_equal is False:raise ValueError('Same codes but decoder noise draws changed')
  if not used_guidance:raise ValueError('Failed to trace actual decoder model guidance')
  output=out/(v['id']+'.wav');sf.write(output,array,48000,subtype='FLOAT');row=dict(v,effective_guidance_scales=used_guidance,guidance_route='per_instance_inference_codes_override' if v.get('override_guidance',False) else 'native_code2sound_argument',model='Muse0.6B+MuCodec',family=r.get('family'),arm=r.get('arm'),seed=r['seed'],decoder_seed=a.seed,token_file_sha256=r['token_file_sha256'],audio_tokens_sha256=key,audio_token_count=len(audio),exact_token_control_group=key,same_decoder_random_draws_as_first_identical_codes=noise_equal,decoder_random_draws=noise,path=str(output),audio_sha256=sha(output),sample_rate=48000,channels=array.shape[1],duration_s=len(array)/48000,inference_seconds=seconds,gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),rms_dbfs=float(20*np.log10(max(np.sqrt(np.mean(array.astype(float)**2)),1e-12))),peak=float(np.max(np.abs(array))),out_of_range_fraction=float(np.mean(np.abs(array)>1)),human_rating=None)
  rows.append(row);(out/'run.json').write_text(json.dumps(dict(status='running',profile=profile,samples=rows),ensure_ascii=False,indent=2,allow_nan=False)+'\n');print(json.dumps({k:row[k] for k in ['id','duration_s','inference_seconds','same_decoder_random_draws_as_first_identical_codes','audio_sha256']}),flush=True)
 (out/'run.json').write_text(json.dumps(dict(status='complete',profile=profile,samples=rows),ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
 l=sub.add_parser('lm');l.add_argument('--model',type=Path,required=True);l.add_argument('--batch',type=Path,required=True);l.add_argument('--output',type=Path,required=True);l.add_argument('--audio-tokens',type=int,default=750)
 d=sub.add_parser('decode');d.add_argument('--mucodec',type=Path,required=True);d.add_argument('--tokens',type=Path,required=True);d.add_argument('--plan',type=Path,required=True);d.add_argument('--output',type=Path,required=True);d.add_argument('--seed',type=int,default=42);d.add_argument('--steps',type=int,default=20);d.add_argument('--duration',type=float,default=40.96)
 a=p.parse_args();lm(a) if a.mode=='lm' else decode(a)
if __name__=='__main__':main()
