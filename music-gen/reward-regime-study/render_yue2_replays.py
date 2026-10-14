"""Native 30-60-second paired replays, separate from full validation statistics."""
import argparse
import json
import subprocess
from pathlib import Path
import soundfile as sf
from peft import PeftModel
import train_yue2 as training
from rewards import sha, score_remote, objective
from yue2 import YuE2Pipeline


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--vae',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--scorer',default='http://127.0.0.1:43140')
    a=p.parse_args()
    if sha(a.data)!=training.DATA_SHA:raise ValueError('Frozen data changed')
    rows=json.loads(a.data.read_text())['valid'];assert len(rows)==30
    training.SAMPLING={**training.SAMPLING,'min_tokens':750,'max_tokens':1500}
    common=a.root/'replays'/'yue2-baseline'
    adapter_model=None;adapter_names={}
    pipe=YuE2Pipeline.from_pretrained(a.model,vae=a.vae,device='cuda',backend='torch-eager',memory_budget_gib=32,progress=False)
    base=pipe._load_model().eval()
    arms=[('baseline',0,None)]
    for reward in ['richness','instrument','arrangement']:
        output=a.root/f'yue2-{reward}-lr3e4-seed42'
        if not (output/'completion.json').exists():raise ValueError('Training must finish before replay stage')
        arms.extend((reward,step,output/'checkpoints'/f'step_{step:06d}'/'adapter') for step in [50,100,270])
    manifest=[]
    for reward,step,adapter in arms:
        if adapter:
            name=f'{reward}-{step}'
            if adapter_model is None:
                adapter_model=PeftModel.from_pretrained(base,adapter,adapter_name=name).eval()
            elif name not in adapter_names:
                adapter_model.load_adapter(adapter,adapter_name=name)
            adapter_names[name]=True;adapter_model.set_adapter(name)
        for index in [0,10,20]:
            folder=(common if not adapter else a.root/'replays'/f'yue2-{reward}'/f'{step:06d}')/f'{index:03d}'
            if not (folder/'rollout.json').exists():training.sample(pipe,rows[index],5101+index,folder)
            metadata=json.loads((folder/'rollout.json').read_text())
            if sha(folder/'audio.flac')!=metadata['audio_sha256']:raise ValueError('Audio changed')
            sr=sf.info(folder/'audio.flac').samplerate
            mp3=folder/'audio.mp3'
            if not mp3.exists():
                subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(folder/'audio.flac'),'-codec:a','libmp3lame','-b:a','192k','-ar','48000',str(mp3)],check=True)
            manifest.append({'reward':reward,'step':step,'valid_index':index,'seed':5101+index,
                'original_sha256':metadata['audio_sha256'],'mp3_sha256':sha(mp3),
                'seconds':metadata['seconds'],'source_sample_rate':sr,'sampling':training.SAMPLING,
                'adapter_sha256':sha(adapter/'adapter_model.safetensors') if adapter else None,
                'truncated':metadata['truncated'],'scope':'Three fixed validation listening examples, not full benchmark means',
                'normalization':False,'source_path':str(folder/'audio.flac'),'mp3_path':str(mp3)})
        training.offload(pipe)
        results=score_remote(a.scorer,[Path(r['source_path']) for r in manifest[-3:]],reward if adapter else 'richness')
        for entry,result in zip(manifest[-3:],results):
            entry['score']=result
            entry['all_objectives']={key:objective(result,key) for key in ['richness','instrument','arrangement']}
        training.write(a.root/'replays'/'private-manifest.json',{'rows':manifest,'data_sha256':training.DATA_SHA})
    training.write(a.root/'replays'/'completion.json',{'examples_per_checkpoint':3,'paired_outputs':len(manifest),'public_format':'MP3'})


if __name__=='__main__':main()
