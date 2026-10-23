"""Stage immutable existing SongEval validation audio for actual SongBench scoring."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
rows=[]
for step in [0,50,100,270]:
    receipt=a.source/f'step_{step:06d}/receipt.json'
    source=json.loads(receipt.read_text())['heldout']
    assert sorted(r['index'] for r in source)==list(range(30))
    for r in source:
        path=a.source/r['audio'];assert sha(path)==r['audio_sha256']
        dest=Path('audio')/f'{step:06d}'/f'{r["index"]:03d}.flac'
        (a.output/dest).parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,a.output/dest)
        rows.append(dict(step=step,index=r['index'],seed=r['seed'],split='valid',
                         audio=str(dest),audio_sha256=r['audio_sha256'],seconds=r['seconds'],
                         source_receipt_sha256=sha(receipt)))
manifest=dict(model='YuE2',trained_reward='SongEval mean5',lr=.0003,
              evaluated_reward='SongBench mean7',n_per_step=30,steps=[0,50,100,270],
              split='valid',source_experiment_sha256=sha(a.source/'experiment.json'),
              source='Existing immutable full validation, not newly generated or test',rows=rows)
(a.output/'input.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(rows=len(rows),manifest_sha256=sha(a.output/'input.json'))))
