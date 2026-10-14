"""Hash-bound MP3 exports and explicitly selected validation counterexamples."""
import argparse
import datetime
import json
import shutil
import subprocess
from pathlib import Path
import numpy as np
from rewards import sha, objective, score_remote


def read(path):return json.loads(path.read_text())


def metrics(row,reward):
    return {'objective':objective(row,reward),'SP_Richness':row['richness']['raw'],
            **row['songbench'],'clipping_fraction':row['guard']['clipping_fraction'],
            'rms_dbfs':row['guard']['rms_dbfs'],'penalty':row['guard']['penalty'],'invalid':row['guard']['invalid']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--scorer',default='http://127.0.0.1:43140');a=p.parse_args()
    out=a.root/'publication';(out/'audio').mkdir(parents=True,exist_ok=True)
    exports={};baselines={};long=[];bad=[];audits=[];validation_archive=[]
    def asset(path,record,reward):
        original=sha(path)
        if original!=record['audio_sha256']:raise ValueError('Score/audio identity mismatch')
        name=f'regime_{original[:24]}.mp3';destination=out/'audio'/name
        if not destination.exists():
            subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(path),'-map','0:a:0',
                '-codec:a','libmp3lame','-b:a','192k','-ar','48000','-map_metadata','-1',str(destination)],check=True)
        exports[name]={'sha256':sha(destination),'source_audio_sha256':original,'bytes':destination.stat().st_size,
                       'sample_rate':48000,'bitrate_kbps':192,'normalization':False}
        return {'audio':'audio/'+name,**exports[name],'seconds':record['seconds'],'metrics':metrics(record,reward)}
    long_manifest=a.root/'replays/private-manifest.json'
    if long_manifest.exists():
        saved=read(long_manifest)
        for r in saved['rows']:
            path=Path(r['source_path']);cache=out/f'score-{r["original_sha256"]}.json'
            if cache.exists():score=read(cache)
            else:
                score=score_remote(a.scorer,[path],r['reward'] if r['reward']!='baseline' else 'richness')[0]
                cache.write_text(json.dumps(score,allow_nan=False)+'\n')
            entry={'reward':r['reward'],'step':r['step'],'index':r['valid_index'],'seed':r['seed'],
                   'sampling':r['sampling'],'truncated':r['truncated'],'adapter_sha256':r['adapter_sha256'],
                   'asset':asset(path,score,r['reward'] if r['reward']!='baseline' else 'richness')}
            if r['reward']=='baseline':baselines[r['valid_index']]=entry
            else:long.append(entry)
    for reward in ['richness','instrument','arrangement']:
        run=a.root/f'yue2-{reward}-lr3e4-seed42'
        valid=[]
        for step in [50,100,270]:
            if (run/f'valid_{step:06d}/receipt.json').exists():valid.append(step)
        if not valid:continue
        for checkpoint in [0,*valid]:
            receipt=run/f'valid_{checkpoint:06d}/receipt.json';data=read(receipt)
            assert data['n']==len(data['rows'])==30
            validation_archive.append({'reward':reward,'step':checkpoint,'n':30,'receipt_sha256':sha(receipt),
                'rows':[{'index':i,'seed':5101+i,'source_audio_sha256':r['audio_sha256'],'seconds':r['seconds'],
                         'metrics':metrics(r,reward)} for i,r in enumerate(data['rows'])]})
        step=max(valid);folder=run/f'valid_{step:06d}';before=read(run/'valid_000000/receipt.json');after=read(folder/'receipt.json')
        assert before['n']==after['n']==len(before['rows'])==len(after['rows'])==30
        candidates=[];deltas=[]
        for i,(b,c) in enumerate(zip(before['rows'],after['rows'])):
            mb,mc=metrics(b,reward),metrics(c,reward)
            delta={k:mc[k]-mb[k] for k in ['objective','SP_Richness','Instrumental','Arrangement','Musicality','Mixing']}
            deltas.append(delta)
            cross=['Instrumental','Arrangement'] if reward=='richness' else ['Musicality','Arrangement' if reward=='instrument' else 'Instrumental']
            if delta['objective']>0 and all(delta[k]<0 for k in cross):
                candidates.append((sum(-delta[k] for k in cross),i,delta,b,c))
        rng=np.random.default_rng(20261014);indices=rng.integers(0,30,(10000,30))
        statistics={}
        for key in deltas[0]:
            values=np.array([d[key] for d in deltas]);bounds=np.quantile(values[indices].mean(1),[.025,.975])
            statistics[key]={'mean_delta':float(values.mean()),'ci95':list(map(float,bounds))}
        audits.append({'reward':reward,'step':step,'n':30,'complete_arm':(run/'completion.json').exists(),
            'candidate_n':len(candidates),'statistics':statistics,
            'selection':'positive guarded target delta AND negative deltas in both '+', '.join(cross)+'; rank by sum of these drops, show top2',
            'interpretation':'Selected proxy-conflict examples, not random samples or confirmed audible reward hacking. CIs resample30 validation conditions, one training seed.'})
        for _,i,delta,b,c in sorted(candidates,reverse=True)[:2]:
            bp=run/f'valid_000000/{i:03d}/audio.flac';cp=folder/f'{i:03d}/audio.flac'
            bad.append({'reward':reward,'step':step,'index':i,'seed':5101+i,'delta':delta,
                'adapter_sha256':sha(run/f'checkpoints/step_{step:06d}/adapter/adapter_model.safetensors'),
                'scope':'Exact short validation audio scored by reward; not the long replay cohort; selected, not representative',
                'baseline':asset(bp,b,reward),'candidate':asset(cp,c,reward)})
    report={'schema':1,'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'model':'yue2','lr':3e-4,'seed':42,'baseline':list(baselines.values()),'long_replays':long,
        'badcases':bad,'audits':audits,'audio_files':exports,'validation_receipts':validation_archive,
        'visual_protocol':{'wave_bounds':[-1,1],'mel_hz':[0,24000],'mel_db':[-80,0],'background':'#000000','visual_source':'decoded public MP3; objective metrics use private lossless originals'}}
    (out/'recovery.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'long':len(long),'baselines':len(baselines),'badcases':len(bad),'files':len(exports),'audits':audits}),flush=True)


if __name__=='__main__':main()
