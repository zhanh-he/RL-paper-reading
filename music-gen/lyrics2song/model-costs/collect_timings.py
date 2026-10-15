"""Read existing complete timing receipts; never generate or score audio."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host',choices=['5090','gadi'],required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    rows={}
    if a.host=='5090':
        path=Path('/storage/zhanh_storage/backbone-rl-20261015/diff-formal/diffrhythm2-richness-lr3e4-beta5e2-seed42/valid_000000/receipt.json')
        data=json.loads(path.read_text())
        if data['n']!=30 or len(data['samples'])!=30:
            raise ValueError('Only complete30-condition baseline accepted')
        measured=[]
        for sample,score in zip(data['samples'],data['rows']):
            if sample['sha256']!=score['audio_sha256']:
                raise ValueError('Scored audio changed')
            measured.append({key:sample[key] for key in ['seconds','generation_seconds','seed','sha256']})
        rows['diffrhythm2']=dict(device='RTX5090 32GB',kind='measured',source=str(path),source_sha256=digest(path),
            scope='Native32-step ODE, CFG2; same zero adapter,30 fixed valid inputs; includes style embedding, transfers, decoder and private FLAC write; excludes scorer, training and model initialization',rows=measured)
        path=Path('/home/mengh/research/YuE2-posttrain/outputs/smoke_20260929/result.json')
        data=json.loads(path.read_text())
        rows['yue2-short-archive']=dict(device='RTX5090 32GB',kind='measured-short-cold-reference',
            source=str(path),source_sha256=digest(path),timing=data['timing'],
            rows=[dict(seconds=data['audio_seconds'],generation_seconds=data['timing']['e2e_seconds'])],
            scope='Historical10.24s truncated setup sample; not a60/300s benchmark and not new torch-eager setup')
    else:
        root=Path('/scratch/iv96/hm6416/open-music-rl-20261014/outputs/heartmula-richness-3e-4/train_rollouts')
        measured=[]
        for path in sorted(root.glob('*/*/rollout.json')):
            data=json.loads(path.read_text())
            measured.append(dict(seconds=data['seconds'],generation_seconds=data['generation_seconds'],
                seed=data['seed'],sha256=data['audio_sha256'],source=str(path),source_sha256=digest(path)))
        if not measured:
            raise ValueError('No historical HeartMuLa timings')
        rows['heartmula']=dict(device='NVIDIA H200 / Gadi gpuhopper',kind='measured-rollout-proxy',
            job_id='181262219.gadi-pbs',exec_host='gadi-gpu-h200-0029',rows=measured,
            scope='Historical native60s-budget head-only rollouts, not new backbone training speed. Includes native global/local generation, HeartCodec, action/logp collection and policy-record save; excludes scoring, updates and initialization. Native precision from archived backend: BF16 generator, FP32 codec; global action distribution FP32, CFG1.5, local topk50.')
    a.output.write_text(json.dumps(dict(host=a.host,scope='Timing evidence only, no reward means or training claims',measurements=rows),indent=2)+'\n')
    print({key:len(value['rows']) for key,value in rows.items()},flush=True)


if __name__=='__main__':
    main()
