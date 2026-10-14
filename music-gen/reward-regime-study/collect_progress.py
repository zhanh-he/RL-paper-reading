"""Export aggregate, verified progress; never export private prompts or audio."""
import argparse
import datetime
import json
from pathlib import Path
from rewards import sha


def collect(root):
    runs = []
    for reward in ['richness', 'instrument', 'arrangement']:
        folder = root/f'yue2-{reward}-lr3e4-seed42'
        records = [json.loads(line) for line in (folder/'steps.jsonl').read_text().splitlines()] if (folder/'steps.jsonl').exists() else []
        assert [r['step'] for r in records] == list(range(1, len(records)+1))
        verified = 0
        for r in records:
            for i,component in enumerate(r['components']):
                assert sha(folder/'train_rollouts'/f'{r["step"]:06d}'/str(i)/'audio.flac') == component['audio_sha256']
                assert component['reward'] == reward
            if (folder/'checkpoints'/f'step_{r["step"]:06d}'/'adapter'/'adapter_model.safetensors').exists():
                verified = r['step']
            else:
                break
        trace = [{'step':r['step'], 'reward':r['reward'], 'kl':r['sampled_kl']} for r in records[:verified]]
        validation = []
        for step in [0,50,100,270]:
            receipt = folder/f'valid_{step:06d}'/'receipt.json'
            if not receipt.exists():
                continue
            data = json.loads(receipt.read_text())
            assert data['n'] == len(data['rows']) == 30
            for i, row in enumerate(data['rows']):
                assert sha(receipt.parent/f'{i:03d}'/'audio.flac') == row['audio_sha256']
            keys=['Arrangement','Instrumental','Mixing','Musicality']
            cross={key:sum(row['songbench'][key] for row in data['rows'])/30 for key in keys}
            cross['SP_Richness']=sum(row['richness']['raw'] for row in data['rows'])/30
            validation.append({'step':step,'n':30,'mean':data['mean_reward'], 'receipt_sha256':sha(receipt),
                               'cross_means':cross,'invalid_n':sum(row['guard']['invalid'] for row in data['rows'])})
        runs.append({'model':'yue2','reward':reward,'steps':verified,'trace':trace,'validation':validation,
                     'complete':(folder/'completion.json').exists() and verified==270})
    return {'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'runs':runs,
            'scope':'Training trace plus all-30 validation receipts. Not WildSongBench test or human MOS.'}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    print(json.dumps(collect(parser.parse_args().root),indent=2,allow_nan=False))
