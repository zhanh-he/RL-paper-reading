"""Use the established all-audio formal collector; retain paired rows for figures."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import soundfile as sf

from collect_formal_run import collect, digest

root=Path(sys.argv[1]);results=[]
for lr in ['2e-5','1e-4','3e-4']:
    folder=root/f'cmi_triple_9to1_songeval_lr{lr}_train_20261007_lab5090'
    record=collect(folder,'songeval',float(lr))
    assert record['status']=='verified-complete'
    record.update(id=f'yue2-songeval-{lr}',lr=float(lr),reward='SongEval mean5',model='YuE2',scope='CMI270/30; short600-token validation, beta0.01 rank4 q/v; not human quality or full-song test')
    for step,point in record['validation'].items():
        path=folder/f'step_{int(step):06d}/receipt.json';receipt=json.loads(path.read_text())
        point['receipt_sha256']=digest(path)
        point['rows']=[dict(index=e['index'],seed=e['seed'],value=e['reward']['mean'],
            dimensions=e['reward'],audio_sha256=e['audio_sha256'],scored_audio_sha256=e['scored_audio_sha256'],
            signal=e['signal']) for e in receipt['heldout']]
        if step=='0':
            for row,entry in zip(point['rows'],receipt['heldout']):
                pcm,sr=sf.read(folder/entry['scored_audio'],dtype='float32',always_2d=True)
                row['pcm_sha256']=hashlib.sha256(pcm.tobytes()).hexdigest()
                row['sample_rate']=sr;row['frames']=len(pcm)
    results.append(record)
print(json.dumps(dict(checked_at=datetime.now(timezone.utc).isoformat(),runs=results),allow_nan=False))
