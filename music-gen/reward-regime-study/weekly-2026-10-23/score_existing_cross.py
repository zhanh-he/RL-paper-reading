"""Full-checkpoint scoring; partial receipts never become a final split mean."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
from types import SimpleNamespace
from rewards import sha,score_remote
from reward_server import Scorer


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


p=argparse.ArgumentParser(description=__doc__)
for key in ['input','output']:p.add_argument('--'+key,type=Path,required=True)
for key in ['bundle','deps']:p.add_argument('--'+key,type=Path)
p.add_argument('--scorer',help='Reuse an existing serialized SongBench service, loading no extra GPU model')
a=p.parse_args();manifest=json.loads(a.input.read_text());rows=manifest['rows']
assert len(rows)==120 and manifest['n_per_step']==30
assert sorted({r['step'] for r in rows})==[0,50,100,270]
assert len({(r['step'],r['index']) for r in rows})==120
for row in rows:assert sha(a.input.parent/row['audio'])==row['audio_sha256']
if not a.scorer and (not a.bundle or not a.deps):p.error('Provide a scorer URL or bundle + deps')
scorer=None if a.scorer else Scorer(SimpleNamespace(bundle=a.bundle,deps=a.deps,mean7_only=True,
                            separator=a.bundle/'unused-separator',receipt_dir=a.output/'scorer'))
identity=None
groups=defaultdict(list)
for index,row in enumerate(rows):
    path=(a.input.parent/row['audio']).resolve()
    target=a.output/f'items/{row["step"]:06d}-{row["index"]:03d}.json'
    if target.exists():
        result=json.loads(target.read_text())
        assert result['audio_sha256']==row['audio_sha256']
    else:
        result=(score_remote(a.scorer,[path],'mean7') if a.scorer else scorer.score([path],'mean7'))[0]
        result.update(index=row['index'],step=row['step'],source_receipt_sha256=row['source_receipt_sha256'])
        write(target,result)
    assert result['scorer_identity']['songbench_weights_sha256']=='ca27966a6d2b3312587b9e1d53ee3adf6bde8ac68a7d339e6f962206cbfde312'
    assert result['scorer_identity']['muq_weights_sha256']=='273febab2be02872c37d2c37e48a9d6c52c1c9392f3eeeabd498efa281ccb7a6'
    assert result['scorer_identity']['mean7_only']
    if identity is None:identity=result['scorer_identity']
    assert result['scorer_identity']==identity
    groups[row['step']].append(result)
    write(a.output/'progress.json',dict(state='scoring',completed=index+1,target=120,
        complete_steps=sorted(s for s,g in groups.items() if len(g)==30),
        updated_at=datetime.now(timezone.utc).isoformat()))
    if len(groups[row['step']])==30:
        scores=groups[row['step']]
        write(a.output/f'valid_{row["step"]:06d}/receipt.json',dict(step=row['step'],n=30,
            rows=scores,mean7=sum(r['mean'] for r in scores)/30,split='valid',
            scope='Full frozen30 valid; actual SongEval-trained checkpoint evaluated by SongBench',
            input_sha256=sha(a.input)))
write(a.output/'completion.json',dict(status='complete',n=120,n_per_step=30,
      steps=[0,50,100,270],trained_reward='SongEval mean5',evaluated_metric='SongBench mean7',
      input_sha256=sha(a.input),updated_at=datetime.now(timezone.utc).isoformat()))
