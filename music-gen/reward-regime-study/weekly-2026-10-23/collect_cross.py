"""Verify all120 completed cross-scores and retain their own archived baseline."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

HEADS=['Vocal','Instrumental','Melody','Structure','Arrangement','Mixing','Musicality']
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--input',type=Path,required=True);p.add_argument('--results',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();manifest=json.loads(a.input.read_text());root=a.results
complete=json.loads((root/'completion.json').read_text())
assert complete['n']==120 and complete['input_sha256']==sha(a.input)
source={(r['step'],r['index']):r for r in manifest['rows']}
validation=[]
identity=None
for step in [0,50,100,270]:
    path=root/f'valid_{step:06d}/receipt.json';receipt=json.loads(path.read_text())
    assert receipt['n']==len(receipt['rows'])==30
    assert sorted(r['index'] for r in receipt['rows'])==list(range(30))
    rows=[]
    for row in receipt['rows']:
        original=source[(step,row['index'])]
        assert row['audio_sha256']==original['audio_sha256']==sha(a.input.parent/original['audio'])
        if identity is None:identity=row['scorer_identity']
        assert row['scorer_identity']==identity and identity['mean7_only']
        assert identity['songbench_weights_sha256']=='ca27966a6d2b3312587b9e1d53ee3adf6bde8ac68a7d339e6f962206cbfde312'
        value=sum(row['songbench'][h] for h in HEADS)/7
        assert abs(row['mean']-value)<1e-12
        rows.append(dict(index=row['index'],audio_sha256=row['audio_sha256'],seconds=row['seconds'],
                         songbench=row['songbench'],mean7=value,source_receipt_sha256=original['source_receipt_sha256']))
    mean=sum(r['mean7'] for r in rows)/30
    assert abs(receipt['mean7']-mean)<1e-12
    validation.append(dict(step=step,n=30,rows=rows,means={'SongBench_mean7':mean},receipt_sha256=sha(path)))
record=dict(status='verified-complete',checked_at=datetime.now(timezone.utc).isoformat(),
    id='yue2-songeval3e4-cross-songbench',model='yue2',reward='songeval',experiment={'lr':.0003,'beta':.01},
    validation=validation,steps_sha256=manifest['source_experiment_sha256'],input_sha256=sha(a.input),
    source_experiment_sha256=manifest['source_experiment_sha256'],scorer_identity=identity,
    binding={'bound_audio_n':120},baseline_protocol='SongEval archive: own frozen baseline, not the SP baseline',
    scope='Actual SongEval-trained YuE2, scored by frozen SongBench on all30 valid x4 checkpoints. Different archived sampling baseline; no matched-reward causal comparison.')
a.output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
print(json.dumps({'step_means':[(r['step'],r['means']['SongBench_mean7']) for r in validation]}))
