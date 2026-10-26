"""CPU transcode of the fixed five scored examples at all four exact checkpoints."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
PUBLIC=ROOT/'platform/site/demos/audio/weekly-20261023'
study='/storage/zhanh_storage/songbench-priority-20261020/outputs'
data=json.loads((HERE/'summary.json').read_text())
old=json.loads((HERE/'listening.json').read_text())
if 'runs' not in old and not (HERE/'listening-20261023-archive.json').exists():
 (HERE/'listening-20261023-archive.json').write_text(json.dumps(old,indent=2)+'\n')
rows=[];runs=[]
program="""from pathlib import Path
import hashlib,subprocess,sys
p=Path(sys.argv[1])
assert hashlib.sha256(p.read_bytes()).hexdigest()==sys.argv[2]
subprocess.run(['ffmpeg','-v','error','-i',str(p),'-ar','48000','-c:a','libmp3lame','-b:a','192k','-f','mp3','pipe:1'],stdout=sys.stdout.buffer,check=True)
"""
for run in data['runs']:
 if (run['model'],run['reward']) not in [('muse','richness'),('muse','mean7'),('diffrhythm2','mean7')]:continue
 rate='1e-4' if run['experiment']['lr']==.0001 else '3e-4'
 runs.append(dict(id=run['id'],model=run['model'],reward=run['reward'],lr=run['experiment']['lr'],
                  steps=[v['step'] for v in run['validation']],scope=run['scope']))
 for point in run['validation']:
  step=point['step']
  for index in [0,10,20,1,2]:
   item=point['rows'][index]
   if run['model']=='muse':
    folder=f'{study}/shared-muse-baseline' if step==0 else f'{study}/{run["id"]}/valid_{step:06d}'
    source=f'{folder}/{index:03d}/audio.flac'
   else:
    parent='diff-mean7-formal' if rate=='3e-4' else 'diff-mean7-lr1e4-formal-20261023'
    source=f'{study}/{parent}/{run["id"]}/valid_{step:06d}/{index:03d}.flac'
   existing=next((r for r in old['rows'] if r['source_audio_sha256']==item['audio_sha256']),None)
   name=(existing['url'].split('/')[-1] if existing else
         f'{run["model"]}-{run["reward"]}-{rate}-step{step}-{index:03d}-{item["audio_sha256"][:8]}.mp3')
   destination=PUBLIC/name
   if not destination.exists():
    temporary=destination.with_suffix('.mp3.tmp')
    with temporary.open('wb') as target:
     subprocess.run(['ssh','lab5090','python3','-c',shlex.quote(program),shlex.quote(source),item['audio_sha256']],stdout=target,check=True)
    temporary.replace(destination)
   assert destination.stat().st_size>10000
   digest=hashlib.sha256(destination.read_bytes()).hexdigest()
   if existing:assert digest==existing['public_mp3_sha256']
   rows.append(dict(run_id=run['id'],model=run['model'],reward=run['reward'],lr=run['experiment']['lr'],step=step,index=index,
                    seconds=item['seconds'],mean7=item['mean7'],source_audio_sha256=item['audio_sha256'],public_mp3_sha256=digest,
                    url='./audio/weekly-20261023/'+name))
record=dict(scope='Four complete native runs; fixed valid0/10/20/1/2 at0/50/100/270. Existing scored audio only; no inference or loudness normalization.',runs=runs,rows=rows)
for path in [HERE/'listening.json',ROOT/'platform/site/demos/weekly-2026-10-23/listening.json']:
 path.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
print(f'Published{len(rows)} scored example bindings as MP3 only;{len({r["url"] for r in rows})} distinct public files; no GPU inference.')
