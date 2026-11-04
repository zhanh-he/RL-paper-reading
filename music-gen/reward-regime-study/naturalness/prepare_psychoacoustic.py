"""Add matched probe groups without replacing the first audit's 25 waveforms.
All probe files stay private; constructed operations are NOT listener labels.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy import signal
from signal_features import sha256


def bandlimited_saw(t,f,limit=16000):
    return -2/np.pi*sum(np.sin(2*np.pi*h*f*t)/h for h in range(1,int(limit/f)+1))


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--directory',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    original=json.loads(Path(a.manifest).read_text())['samples']; rows=list(original)
    dest=Path(a.directory).resolve();dest.mkdir(parents=True,exist_ok=True);sr=48000
    def write(rid,x,metadata):
        path=dest/(rid+'.wav');sf.write(path,x,sr,subtype='FLOAT')
        if metadata.get('source_original_sha256')=='0'*64:metadata['source_original_sha256']=sha256(path)
        rows.append(metadata|dict(id=rid,path=str(path),excerpt_sha256=sha256(path),duration_s=len(x)/sr,sample_rate=sr))
    for base_id in ['muse_baseline','ref_vibeace']:
        base=next(r for r in original if r['id']==base_id);x,_=sf.read(base['path'],always_2d=True)
        t=np.arange(len(x))/sr;rms=np.sqrt(np.mean(x*x)); rng=np.random.default_rng(20261104)
        tone=np.sin(2*np.pi*4200*t)[:,None]; tone=np.repeat(tone,2,axis=1)
        env=sum(np.where(t>=start,np.exp(-np.maximum(0,t-start)/.25),0) for start in [1.2,3.7,6.2,8.3])[:,None]
        high=x-signal.sosfiltfilt(signal.butter(4,3000,fs=sr,output='sos'),x,axis=0)
        variants={'anchor':x,'am70_mild':x*(1+.25*np.cos(2*np.pi*70*t[:,None])),
                  'am70_deep':x*(1+.8*np.cos(2*np.pi*70*t[:,None])),
                  'tone4200_weak':x+.08*rms*np.sqrt(2)*tone,
                  'tone4200_strong':x+.25*rms*np.sqrt(2)*tone,
                  'ring4200_bursts':x+.55*rms*tone*env,
                  'high_shelf_style':x+high,
                  'comb_style':x+.35*np.roll(x,int(.0012*sr),axis=0)}
        # Equal RMS throughout each group, then ONE common headroom factor.
        variants={k:y*rms/max(np.sqrt(np.mean(y*y)),1e-12) for k,y in variants.items()}
        common=min(1.,.9/max(np.max(abs(y)) for y in variants.values()))
        anchor='psy_'+base_id+'__anchor'
        for kind,y in variants.items():
            rid='psy_'+base_id+'__'+kind
            meta={k:base[k] for k in ['model','checkpoint','input_id','source_original_sha256','rights','credit','source_url','license_url','source_asset'] if k in base}
            meta.update(label=base['label']+' / '+kind,kind='psychoacoustic_probe',base_id=anchor if kind!='anchor' else base_id,
                        pair_group='psy_'+base_id,transform=kind,common_headroom_gain=common,
                        comparison_id=anchor if kind!='anchor' else base_id,
                        selection_reason='Predefined equal-RMS probe group; style controls and induced operations have no perceptual labels',
                        protected_style_control=kind in ['anchor','high_shelf_style','comb_style'])
            write(rid,y*common,meta)
    t=np.arange(10*sr)/sr
    anchors={'tone1k':np.sin(2*np.pi*1000*t),'am1k_4hz':(1+np.cos(2*np.pi*4*t))*np.sin(2*np.pi*1000*t),
             'am1k_70hz':(1+np.cos(2*np.pi*70*t))*np.sin(2*np.pi*1000*t),
             'am1k_200hz':(1+np.cos(2*np.pi*200*t))*np.sin(2*np.pi*1000*t)}
    bell=np.zeros_like(t);bass=np.zeros_like(t)
    for k,f in enumerate([220,277.1826,329.6276,440,369.9944]):
        sel=(t>=k*2)&(t<(k+1)*2);q=t[sel]-k*2
        bell[sel]=np.exp(-q*2)*np.sin(2*np.pi*f*q+2*np.exp(-q)*np.sin(2*np.pi*f*2.7*q))
        bass[sel]=np.minimum(q/.03,1)*np.minimum((2-q)/.08,1)*bandlimited_saw(q,f/2)
    anchors.update(fm_bell_style=bell,saw_bass_style=bass)
    for name,y in anchors.items():
        y=.08*y/max(np.sqrt(np.mean(y*y)),1e-12)
        write('psy_control_'+name,np.column_stack([y,y]),dict(label=name,kind='constructed_control',model='deterministic signal',checkpoint='not applicable',
              input_id='psychoacoustic anchor/style control',pair_group='psy_anchor',transform=name,rights='Original constructed signal; CC0-1.0',
              source_original_sha256='0'*64,selection_reason='Analytical anchor or intended electronic timbre; not human naturalness gold',protected_style_control='style' in name))
    Path(a.output).write_text(json.dumps(dict(schema_version=1,samples=rows),ensure_ascii=False,indent=2)+'\n')
    print(f'{len(rows)} files, first25 unchanged; two equal-RMS 8-variant groups + six controls')


if __name__=='__main__':main()
