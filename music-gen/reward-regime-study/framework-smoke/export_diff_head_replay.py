"""Recover existing head-ablation validation audio, without new inference."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from rewards import sha
from export_public_recovery import metrics

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
(a.output/'audio').mkdir(parents=True,exist_ok=True)
baseline=[];long=[];exports={}
for step in [0,50,100]:
    folder=a.run/f'valid_{step:06d}'
    receipt=json.loads((folder/'receipt.json').read_text())
    assert receipt['n']==len(receipt['rows'])==30
    for index in [0,10,20]:
        score=receipt['rows'][index]
        source=Path(score.get('private_audio',folder/f'{index:03d}'/'audio.flac'))
        digest=sha(source);assert digest==score['audio_sha256']
        name=f'regime_{digest[:24]}.mp3';dest=a.output/'audio'/name
        if not dest.exists():
            subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(source),'-map','0:a:0',
                '-codec:a','libmp3lame','-b:a','192k','-ar','48000','-map_metadata','-1',str(dest)],check=True)
        binding={'sha256':sha(dest),'source_audio_sha256':digest,'bytes':dest.stat().st_size,
                 'sample_rate':48000,'bitrate_kbps':192,'normalization':False}
        exports[name]=binding
        row={'reward':'baseline' if step==0 else 'richness','step':step,'index':index,'seed':5101+index,
             'scope':'Existing native60s-budget full-validation audio; deprecated proj_out-only ablation, not backbone LoRA',
             'asset':{'audio':'audio/'+name,**binding,'seconds':score['seconds'],'metrics':metrics(score,'richness')}}
        (baseline if step==0 else long).append(row)
report={'schema':1,'model':'diffrhythm2','lr':3e-4,'seed':42,'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope':'Deprecated head-only validation replay; do not compare as a completed backbone experiment',
        'baseline':baseline,'long_replays':long,'audio_files':exports,'audits':[],'badcases':[],'validation_receipts':[]}
(a.output/'recovery.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({'baseline':len(baseline),'outputs':len(long),'seconds':[r['asset']['seconds'] for r in baseline+long]}),flush=True)
