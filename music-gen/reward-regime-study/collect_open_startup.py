"""Collect a bounded, verified startup snapshot; this does not monitor a job."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', type=Path, required=True)
p.add_argument('--output', type=Path, required=True); p.add_argument('--steps', type=int, default=5); a=p.parse_args()
rows=[]
with (a.run/'steps.jsonl').open() as stream:
    for line in stream:
        if len(rows) == a.steps: break
        rows.append(json.loads(line))
assert len(rows)==a.steps and [r['step'] for r in rows]==list(range(1,a.steps+1))
for row in rows:
    checkpoint=a.run/'checkpoints'/f'step_{row["step"]:06d}.pt'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==row['checkpoint_sha256']
report={'model':'diffrhythm2','reward':'richness','steps':len(rows),'complete':False,'validation':[],
        'trace':[{'step':r['step'],'reward':r['reward'],'kl':r['sampled_kl']} for r in rows],
        'snapshot_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'kl_unit':'Exact conditional Gaussian KL per latent dimension, averaged over sampled transitions',
        'scope':'Verified first5 updates only; not a final score or full validation mean'}
a.output.mkdir(parents=True,exist_ok=True)
(a.output/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
(a.output/'steps.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
(a.output/'experiment.json').write_bytes((a.run/'experiment.json').read_bytes())
print(json.dumps({'verified_steps':len(rows),'full_valid_n':0,'complete':False}))
