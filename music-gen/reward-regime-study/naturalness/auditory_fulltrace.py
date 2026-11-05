"""GPU auditory descriptors and unmodified CPU MoSQITo reference roughness.

Custom descriptors are NOT standardized roughness, audibility probabilities or
validated naturalness rewards. Digital levels have an explicitly assumed SPL.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import time
import numpy as np
from scipy import signal
import soundfile as sf
import torch
from mosqito.sq_metrics import roughness_dw


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(x):
    x=np.asarray(x,dtype=float)
    if not x.size or not np.isfinite(x).all():raise ValueError('Nonfinite/empty feature')
    return dict(mean=float(x.mean()),p50=float(np.median(x)),p95=float(np.quantile(x,.95)),maximum=float(x.max()))


class AuditoryGPU:
    def __init__(self,device='cuda',sr=48000):
        self.device=device;self.sr=sr
        erb=lambda f:21.4*np.log10(1+.00437*f)
        self.centers=(10**(np.linspace(erb(150),erb(16000),32)/21.4)-1)/.00437
        self.filters=torch.tensor(np.stack([signal.gammatone(float(f),'fir',order=4,numtaps=4096,fs=sr)[0] for f in self.centers]),dtype=torch.float32,device=device)
        self.lowpass=torch.tensor(signal.firwin(257,400,fs=sr),dtype=torch.float32,device=device)
    def __call__(self,audio):
        dev=self.device;sr=self.sr;x=torch.as_tensor(audio,dtype=torch.float32,device=dev)
        if x.ndim!=1 or len(x)<sr:raise ValueError('Need mono >=1s')
        rms=float(torch.sqrt(torch.mean(x*x)))
        if rms<1e-5:return dict(valid=False,modulation_depth_20_300=0.,modulation_share_20_300=0.,isolated_resonance_proxy=0.,peak_prominence_p95_db=None,peak_persistence=0.,maps=None)
        n=len(x);fft=2**int(np.ceil(np.log2(n+4096-1)))
        y=torch.fft.irfft(torch.fft.rfft(self.filters,n=fft)*torch.fft.rfft(x,n=fft),n=fft)[:,:n]
        # Analytic signal; FFT is zero-padded beyond the convolution support.
        spectrum=torch.fft.fft(y,n=fft);h=torch.zeros(fft,device=dev);h[0]=1;h[1:fft//2]=2;h[fft//2]=1
        env=torch.abs(torch.fft.ifft(spectrum*h,dim=-1))[:,:n]
        # Low-pass before 48x decimation; retain 400Hz of envelope bandwidth.
        env=torch.fft.irfft(torch.fft.rfft(env,n=fft)*torch.fft.rfft(self.lowpass,n=fft),n=fft)[:,::48]
        env=env[:,100:int(n/48)-100] # trim startup/end filter transients
        blocks=env.unfold(1,500,250);mu=blocks.mean(-1)
        active=mu>mu.max()*1e-3
        w=torch.hann_window(500,device=dev);power=torch.abs(torch.fft.rfft((blocks-mu[...,None])*w,dim=-1))**2
        fm=torch.fft.rfftfreq(500,d=.001,device=dev);fast=(fm>=20)&(fm<=300);allmod=(fm>=2)&(fm<=300)
        # Parseval variance estimate with Hann-energy correction, divided by DC.
        var=2*power[...,fast].sum(-1)/(500*(w*w).sum())
        depth=torch.sqrt(var.clamp_min(0))/mu.clamp_min(1e-12)
        ratio=power[...,fast].sum(-1)/power[...,allmod].sum(-1).clamp_min(1e-20)
        weights=mu*mu*active; pertime=(depth*weights).sum(0)/weights.sum(0).clamp_min(1e-20)
        mod=float((depth*weights).sum()/weights.sum());share=float((ratio*weights).sum()/weights.sum())
        mod_psd=(power[...,fast]/mu[...,None].square().clamp_min(1e-20)*weights[...,None]).sum((0,1))/weights.sum().clamp_min(1e-20)
        # Concentration in a +/-4Hz modulation-rate neighborhood. Hypothesis:
        # steady buzz is more rate-concentrated than broadband transient changes.
        smooth=torch.nn.functional.conv1d(mod_psd[None,None,:],torch.ones((1,1,5),device=dev),padding=2)[0,0]
        peak_index=int(smooth.argmax());peak_fraction=float(smooth.max()/mod_psd.sum().clamp_min(1e-20))
        # Local narrow-peak prominence. This is a relative spectral heuristic,
        # without a validated masking threshold or multi-F0 assignment.
        z=torch.stft(x,n_fft=4096,hop_length=480,window=torch.hann_window(4096,device=dev),return_complex=True,center=False)
        p=abs(z)**2;db=10*torch.log10(p.clamp_min(1e-20));freq=torch.fft.rfftfreq(4096,1/sr,device=dev)
        background=torch.nn.functional.pad(db.T.unsqueeze(1),(15,15),mode='replicate').unfold(-1,31,1).median(-1).values.squeeze(1).T
        prominence=db-background
        local=(p>torch.roll(p,1,0))&(p>torch.roll(p,-1,0))
        frames=p.sum(0)>p.sum(0).max()*1e-5
        region=(freq>=250)&(freq<16000)
        eligible=(p>p.sum(0,keepdim=True)*1e-5)&frames&local&region[:,None]
        score=((prominence-12)/24).clamp(0,1)*eligible
        persistence=((prominence>12)&eligible).float().sum(1)/frames.sum().clamp_min(1)
        # Max fixed-frequency persistent excess; legitimate partials can score high.
        fixed=(score.sum(1)/frames.sum().clamp_min(1))
        target=int(fixed.argmax()); topframes=prominence[eligible]
        highregion=(freq>=3000)&(freq<8000)
        highscore=float(fixed[highregion].max())
        # Exploratory harmonic veto: can a strong lower spectral peak explain a
        # candidate as an integer partial? No single-F0 assumption on a mixture.
        # This can also hide artifacts and misread inharmonic instruments.
        delta=.5*(torch.roll(db,1,0)-torch.roll(db,-1,0))/(torch.roll(db,1,0)-2*db+torch.roll(db,-1,0)).clamp(max=-1e-9)
        refined=(torch.arange(p.shape[0],device=dev)[:,None]+delta.clamp(-.5,.5))*(sr/4096)
        low=(freq>=80)&(freq<=1500)
        lowerfreq=refined[low];lowergood=local[low]&(p[low]>p.sum(0,keepdim=True)*1e-3)
        bestscore=0.;besttarget=target;bestunexplained=None
        for index in torch.topk(fixed,k=12).indices.tolist():
            fhigh=refined[index][None,:];order=torch.round(fhigh/lowerfreq.clamp_min(1))
            tolerance=torch.maximum(torch.full_like(fhigh,20),fhigh*.003)
            explained=(((fhigh-order*lowerfreq).abs()<tolerance)&(order>=2)&(order<=30)&lowergood).any(0)
            if float(freq[index])<=1500:
                upperorder=torch.round(refined/fhigh.clamp_min(1))
                support=(((refined-upperorder*fhigh).abs()<torch.maximum(torch.full_like(refined,20),refined*.003))
                         &(upperorder>=2)&(upperorder<=15)&local&(p>p.sum(0,keepdim=True)*1e-3)).sum(0)
                explained=explained|(support>=2)
            protected=score[index]*(~explained)
            value=float(protected.sum()/frames.sum().clamp_min(1))
            if value>bestscore:bestscore=value;besttarget=index;bestunexplained=protected
        result=dict(valid=True,modulation_depth_20_300=mod,modulation_share_20_300=share,
                    fast_modulation_peak_fraction=peak_fraction,fast_modulation_peak_hz=float(fm[fast][peak_index]),
                    coherent_modulation_depth=mod*np.sqrt(peak_fraction),
                    modulation_time=stats(pertime.detach().cpu().numpy()),isolated_resonance_proxy=float(fixed.max()),
                    resonance_3_8_proxy=highscore,harmonic_veto_resonance_proxy=bestscore,
                    harmonic_veto_candidate_hz=float(freq[besttarget]) if bestscore>0 else None,
                    peak_prominence_p95_db=float(torch.quantile(topframes,.95)) if topframes.numel() else None,
                    peak_persistence=float(persistence[target]),candidate_frequency_hz=float(freq[target]) if float(fixed.max())>0 else None,
                    resonance_time=dict(time_s=(4096/2/sr+np.arange(score.shape[1])*.01).tolist(),relative_excess=score[target].detach().cpu().numpy().tolist(),
                                        harmonic_veto_excess=bestunexplained.detach().cpu().numpy().tolist() if bestunexplained is not None else [0.] *len(np.arange(score.shape[1]))),
                    maps=dict(center_hz=self.centers.tolist(),time_s=(.1+.25+np.arange(blocks.shape[1])*.25).tolist(),
                              modulation_depth=depth.detach().cpu().numpy().tolist(),
                              band_energy_relative=(mu/mu.max()).detach().cpu().numpy().tolist()))
        return result


def reference_roughness(x,sr,levels):
    started=time.perf_counter()
    rms=float(np.sqrt(np.mean(x*x)))
    if rms<1e-5:return dict(valid=False,levels={},raw_assumed=None,reason='whole-clip silent; 0 is not a naturalness score')
    rows={}
    for spl in levels:
        pa=x*(20e-6*10**(spl/20))/rms
        v,spec,bark,t=roughness_dw(pa,sr,overlap=.5)
        rows[str(spl)]=dict(asper=stats(v),time_s=np.asarray(t).tolist(),time_asper=np.asarray(v).tolist())
        if spl==70:
            rows[str(spl)]['specific_asper']=np.asarray(spec).tolist();rows[str(spl)]['bark_axis']=np.asarray(bark).tolist()
    # Assumption: full-scale sine RMS corresponds to100 dB SPL =>1FS=2.828Pa.
    raw=x*(20e-6*10**(100/20)*np.sqrt(2))
    v,_,_,_=roughness_dw(raw,sr,overlap=.5)
    return dict(valid=True,levels=rows,raw_assumed=dict(asper=stats(v),spl_db=float(20*np.log10(np.sqrt(np.mean(raw*raw))/20e-6))),
                seconds=time.perf_counter()-started,worker_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1024 if platform.system()=='Linux' else 1))


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--audio-dir');p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');p.add_argument('--skip-roughness',action='store_true');p.add_argument('--limit',type=int);p.add_argument('--threads',type=int,default=4);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    torch.set_num_threads(a.threads);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False
    init=time.perf_counter();extract=AuditoryGPU(a.device)
    if a.device=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    initialization=time.perf_counter()-init;rows=[];start=time.perf_counter()
    samples=json.loads(Path(a.manifest).read_text())['samples'];samples=samples[:a.limit] if a.limit else samples
    pool=ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) if not a.skip_roughness else None
    futures=[]
    for r in samples:
        path=Path(a.audio_dir)/(r['id']+'.wav') if a.audio_dir else Path(r['path']);actual=sha(path)
        if actual!=r['excerpt_sha256']:raise ValueError('Audio hash changed '+r['id'])
        x,sr=sf.read(path,dtype='float64',always_2d=True);mono=x.mean(1)
        if sr!=48000 or not np.isfinite(x).all():raise ValueError('Need finite48k')
        begin=time.perf_counter()
        with torch.inference_mode():features=extract(mono)
        if a.device=='cuda':torch.cuda.synchronize()
        gpu_s=time.perf_counter()-begin
        futures.append(pool.submit(reference_roughness,mono,sr,[60,70,80]) if pool else None)
        row=dict(id=r['id'],audio_sha256=actual,auditory=features,roughness=None,
                 timing=dict(auditory_seconds=gpu_s))
        rows.append(row)
        print(json.dumps(dict(id=r['id'],gpu_complete=True,gpu_seconds=gpu_s)),flush=True)
    for row,future in zip(rows,futures):
        roughness=future.result() if future else None;row['roughness']=roughness
        row['timing']['roughness_cpu_seconds']=roughness.get('seconds',0) if roughness else 0
        Path(a.output).write_text(json.dumps(dict(samples=rows,status='running'),allow_nan=False)+'\n')
        print(json.dumps(dict(id=row['id'],roughness70=roughness['levels'].get('70',{}).get('asper') if roughness else None,cpu_s=row['timing']['roughness_cpu_seconds'])),flush=True)
    if pool:pool.shutdown()
    profile=dict(device=torch.cuda.get_device_name() if a.device=='cuda' else 'CPU',torch=torch.__version__,python=platform.python_version(),
                 packages={k:importlib.metadata.version(k) for k in ['mosqito','numpy','scipy','soundfile']},threads=a.threads,cpu_workers=a.workers,blas_threads=os.environ.get('OPENBLAS_NUM_THREADS'),
                 gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated() if a.device=='cuda' else 0,
                 peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1024 if platform.system()=='Linux' else 1),
                 initialization_seconds=initialization,total_seconds=time.perf_counter()-start,manifest_sha256=sha(a.manifest))
    Path(a.output).write_text(json.dumps(dict(schema_version=1,status='complete',profile=profile,samples=rows),allow_nan=False)+'\n')


if __name__=='__main__':main()
