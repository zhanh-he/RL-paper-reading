"""Transcode existing scored originals only. No model invocation or normalization."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
PUBLIC=ROOT/'platform/site/demos/audio/weekly-20261023';PUBLIC.mkdir(parents=True,exist_ok=True)
study='/storage/zhanh_storage/songbench-priority-20261020/outputs'
data=json.loads((HERE/'summary.json').read_text());rows=[]
for run in data['runs']:
    if (run['model'],run['reward']) not in [('muse','richness'),('diffrhythm2','mean7')]:continue
    for step in [0,270]:
        point=next(v for v in run['validation'] if v['step']==step)
        for index in [0,10,20,1,2]:
            item=point['rows'][index]
            if run['model']=='muse':
                folder=f'{study}/shared-muse-baseline' if step==0 else f'{study}/{run["id"]}/valid_000270'
                source=f'{folder}/{index:03d}/audio.flac'
            else:source=f'{study}/diff-mean7-formal/{run["id"]}/valid_{step:06d}/{index:03d}.flac'
            name=f'{run["model"]}-{"baseline" if step==0 else "grpo270"}-{index:03d}.mp3';destination=PUBLIC/name
            if not destination.exists():
                command=f'test "$(sha256sum {shlex.quote(source)} | cut -d" " -f1)" = {item["audio_sha256"]} && ffmpeg -v error -i {shlex.quote(source)} -ar 48000 -c:a libmp3lame -b:a 192k -f mp3 pipe:1'
                temp=destination.with_suffix('.mp3.tmp')
                with temp.open('wb') as target:subprocess.run(['ssh','lab5090',command],stdout=target,check=True)
                temp.replace(destination)
            assert destination.stat().st_size>10000
            rows.append(dict(model=run['model'],step=step,index=index,seconds=item['seconds'],mean7=item['mean7'],
                source_audio_sha256=item['audio_sha256'],public_mp3_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                url='./audio/weekly-20261023/'+name))
record=dict(scope='Fixed five validation indices, native existing10.24s/60s outputs. CPU transcode only; no generated replacement, no normalization.',rows=rows)
for path in [HERE/'listening.json',ROOT/'platform/site/demos/weekly-2026-10-23/listening.json']:
    path.write_text(json.dumps(record,indent=2)+'\n')
print('Exported20 SHA-bound MP3s, zero inference calls')
