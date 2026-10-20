"""CPU-only export of five pre-fixed Diff validation examples; no inference."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
PUBLIC = ROOT/'platform/site/demos/audio/songbench-review-20261020'
data = json.loads((HERE/'verified-recovery.json').read_text())
PUBLIC.mkdir(parents=True, exist_ok=True)
manifest = []
for run in data['runs']:
    if run['model'] != 'diffrhythm2' or run['status'] != 'verified-historical-comparator':
        continue
    reward = run['experiment']['reward']
    for step in ([0, 270] if reward == 'richness' else [270]):
        validation = next(v for v in run['validation'] if v['step'] == step)
        for index in [0, 10, 20, 1, 2]:
            scored = validation['rows'][index]
            source = f'/storage/zhanh_storage/backbone-rl-20261015/diff-formal/{run["run"]}/valid_{step:06d}/{index:03d}.flac'
            name = f'{"baseline" if step == 0 else reward+"-270"}-{index:03d}.mp3'
            destination = PUBLIC/name
            if not destination.exists():
                command = 'ffmpeg -v error -i '+shlex.quote(source)+' -ar 48000 -c:a libmp3lame -b:a 192k -f mp3 pipe:1'
                temporary = destination.with_suffix('.mp3.tmp')
                with temporary.open('wb') as target:
                    subprocess.run(['ssh', 'lab5090', command], stdout=target, check=True)
                temporary.replace(destination)
            assert destination.stat().st_size > 10000
            manifest.append(dict(arm='baseline' if step == 0 else reward, step=step, index=index,
                seconds=scored['seconds'], source_audio_sha256=scored['audio_sha256'],
                public_mp3_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                url='./audio/songbench-review-20261020/'+name, mean7=scored['mean7']))
(HERE/'listening.json').write_text(json.dumps(dict(
    scope='Historical Diff backbone subhead/SP comparison; not SongBench mean7 training. Fixed valid0/10/20/1/2; no score selection. MP3 export only; no new generation.',
    rows=manifest), indent=2)+'\n')
print(f'Exported {len(manifest)} existing scored listening copies; no inference or training')
