"""Freeze all192 WSB prompts x2 fixed seeds in the existing native Muse format."""
import argparse
import json
from pathlib import Path
from rewards import sha
from prepare_muse_cmi_baseline import prompt

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
assert sha(a.source)=='859e5225d319912efb64d587d1d5eb6cf0abc9fb162cc3d5f02d1688cff2edc0'
source=[json.loads(line) for line in a.source.open() if line.strip()]
assert len(source)==192 and [r['prompt_index'] for r in source]==list(range(192))
rows=[]
for r in source:
    for j in range(2):
        rows.append(dict(messages=[dict(role='user',content=prompt(r)),dict(role='assistant',content='')],
                         prompt_index=r['prompt_index'],replicate=j,seed=r['candidate_ar_seeds'][j]))
a.output.mkdir(parents=True,exist_ok=False)
path=a.output/'muse-wsb-test.jsonl'
path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
(a.output/'manifest.json').write_text(json.dumps(dict(source_sha256=sha(a.source),projection_sha256=sha(path),
    conditions=192,replicates=2,rows=384,selection='No test-based selection; all fixed0/50/100/270 checkpoints reported',
    input='Native Muse text+lyrics; no ABC or reference audio'),indent=2)+'\n')
print(json.dumps(dict(rows=384,projection_sha256=sha(path))))
