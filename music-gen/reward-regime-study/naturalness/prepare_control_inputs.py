"""Expand this experiment's public own-text registry into local inference batches."""
import argparse,copy,hashlib,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--registry',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--reference-manifest',help='Original Suno scoring manifest, with id/path/excerpt_sha256');a=p.parse_args();registry=json.loads(Path(a.registry).read_text());out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
 refs={r['id']:r for r in json.loads(Path(a.reference_manifest).read_text())['samples']} if a.reference_manifest else {}
 names={'diff_batch':'diff_batch.json','diff_reference_batch':'diff_reference_batch.json','bloom_batch':'bloom_batch.json','muse_batch':'muse_batch.json','muse_actual_decode_plan':'muse_decode_plan.json'}
 for key,name in names.items():
  obj=copy.deepcopy(registry[key]);needs=any(r.get('reference_id') for r in obj.get('cases',[]))
  if needs and not refs:print('Skipped reference batch until --reference-manifest supplied: '+name);continue
  for case in obj.get('cases',[]):
   if not case.get('reference_id'):continue
   row=refs[case['reference_id']];path=Path(row['path']);actual=hashlib.sha256(path.read_bytes()).hexdigest()
   if actual!=row['excerpt_sha256']:raise ValueError('Rebuilt reference does not match its current manifest')
   case['reference_path']=str(path.resolve());case['reference_sha256']=actual
  (out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');print(name)
if __name__=='__main__':main()
