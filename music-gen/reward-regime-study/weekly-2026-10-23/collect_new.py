"""Read-only, all-audio CPU recovery for the two completed Oct20 priority arms."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

HEADS=['Vocal','Instrumental','Melody','Structure','Arrangement','Mixing','Musicality']
BASE_SHA='c810b2a748579e06e960019f89a9cb98701e4ff545e2c760e8ceaa3343ff1b52'
DATA_SHA='c39d43e81793bee3f312c7028a0483b5da41068db4544acd258115043101e46f'


def load(p):return json.loads(p.read_text())


def sha(p):
    digest=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):digest.update(block)
    return digest.hexdigest()


def verify(score,path,reward,stats):
    assert sha(path)==score['audio_sha256'],'Score/audio binding changed'
    assert score['reward']==reward
    identity=score['scorer_identity']
    assert identity['songbench_weights_sha256']=='ca27966a6d2b3312587b9e1d53ee3adf6bde8ac68a7d339e6f962206cbfde312'
    assert identity['muq_weights_sha256']=='273febab2be02872c37d2c37e48a9d6c52c1c9392f3eeeabd498efa281ccb7a6'
    assert identity['scored_format']=='PCM24-FLAC'
    values=[score['songbench'][key] for key in HEADS]
    assert all(math.isfinite(v) for v in values)
    if reward=='mean7':
        assert identity['mean7_only'] and score['richness']['raw'] is None
        expected=sum(values)/7
    else:
        assert not identity['mean7_only']
        expected=0. if score['guard']['invalid'] else score['richness']['raw']-score['guard']['penalty']
    assert abs(expected-score['mean'])<1e-6
    stats['bound_audio_n']+=1


def point(step,scores,receipt_sha):
    assert len(scores)==30
    rows=[dict(index=i,audio_sha256=r['audio_sha256'],seconds=r['seconds'],songbench=r['songbench'],
        mean7=sum(r['songbench'][key] for key in HEADS)/7,target=r['mean'],
        sp=r['richness']['raw'],guard=r['guard']) for i,r in enumerate(scores)]
    means={key:sum(r['songbench'][key] for r in scores)/30 for key in HEADS}
    means['SongBench_mean7']=sum(r['mean7'] for r in rows)/30
    means['SP_Richness']=sum(r['sp'] for r in rows)/30 if all(r['sp'] is not None for r in rows) else None
    return dict(step=step,n=30,target_mean=sum(r['mean'] for r in scores)/30,means=means,rows=rows,
        invalid_n=sum(r['guard']['invalid'] for r in scores),receipt_sha256=receipt_sha)


def collect(root,kind):
    stats=dict(bound_audio_n=0)
    experiment=load(root/'experiment.json')
    trace=[]
    if kind=='muse':
        assert experiment['data']['train']['n']==270 and experiment['data']['valid']['n']==30
        assert experiment['data']['train']['sha256']=='ff437d751d62742d92daf7ac56a678010c52d6c22def5624a8324301e6699be6'
        assert experiment['data']['valid']['sha256']=='ed5c5f36af48ef6ce81d15c7b15b429fb250ec4a52a04a43e55d96ec3cafb59d'
        identity=load(root/'base-identity.json');assert identity['model_sha256']==BASE_SHA and not identity['starting_from_grpo']
        finals=list((root/'checkpoint').glob('*/checkpoint-270'));assert len(finals)==1
        final=finals[0];state=load(final/'trainer_state.json')
        history=[r for r in state['log_history'] if 'reward' in r]
        assert state['global_step']==270 and [r['step'] for r in history]==list(range(1,271))
        scores=[json.loads(line) for line in (root/'reward-bindings.jsonl').open()]
        assert len(scores)==540
        index={sha(p):p for p in (root/'decoded').glob('*.flac')}
        for score in scores:verify(score,index[score['audio_sha256']],'richness',stats)
        for i,row in enumerate(history):
            assert abs(sum(r['mean'] for r in scores[2*i:2*i+2])/2-row['reward'])<1e-6
            assert math.isfinite(row['kl']) and math.isfinite(row['grad_norm'])
            assert row['learning_rate']==.0003
            trace.append(dict(step=row['step'],reward=row['reward'],kl=row['kl']))
        assert any(r['grad_norm']>0 for r in history)
        baseline=root.parent/'shared-muse-baseline'
        validation=[]
        for step in [0,270]:
            path=root/f'valid_{step:06d}/receipt.json';receipt=load(path)
            assert receipt['n']==len(receipt['rows'])==30
            folder=baseline if step==0 else root/f'valid_{step:06d}'
            for i,score in enumerate(receipt['rows']):
                meta=load(folder/f'{i:03d}/generation.json')
                assert meta['seed']==5101+i and meta['base_model_sha256']==BASE_SHA
                assert meta['audio_sha256']==score['audio_sha256']
                verify(score,folder/f'{i:03d}/audio.flac','richness',stats)
            p=point(step,receipt['rows'],sha(path));assert abs(p['target_mean']-receipt['mean_reward'])<1e-8
            validation.append(p)
        for name in ['optimizer.pt','rng_state.pth','adapter_model.safetensors']:assert (final/name).is_file()
        trace_sha=sha(final/'trainer_state.json')
        scope='Fresh original non-GRPO Muse; ms-swift/TRL backbone LoRA rank8;512 completion tokens; MuCodec max10.24s budget, native shorter clips retained; guarded SP objective.'
    else:
        assert experiment['data_sha256']==DATA_SHA and experiment['train_n']==270 and experiment['valid_n']==30
        rows=[json.loads(line) for line in (root/'steps.jsonl').open()]
        assert [r['step'] for r in rows]==list(range(1,271))
        for row in rows:
            assert row['policy_version']==row['step']-1 and row['beta']==.05 and row['lr']==.0003
            assert row['metrics']['nonzero_attention']>0 and row['metrics']['nonzero_mlp']>0
            for sample,score in zip(row['samples'],row['scores']):
                assert sample['sha256']==score['audio_sha256']
                verify(score,Path(sample['path']),'mean7',stats)
            trace.append(dict(step=row['step'],reward=row['reward'],kl=row['kl']))
        validation=[]
        for step in [0,50,100,270]:
            path=root/f'valid_{step:06d}/receipt.json';receipt=load(path)
            assert receipt['n']==len(receipt['rows'])==30
            assert [r['seed'] for r in receipt['samples']]==list(range(5101,5131))
            for sample,score in zip(receipt['samples'],receipt['rows']):
                assert sample['policy_version']==step and sample['sha256']==score['audio_sha256']
                verify(score,Path(sample['path']),'mean7',stats)
            p=point(step,receipt['rows'],sha(path));assert abs(p['target_mean']-receipt['mean'])<1e-8
            validation.append(p)
        final=root/'checkpoints/iter_0000270'
        for name in ['model','optimizer','lr_scheduler']:assert list((final/name).glob('*.distcp'))
        assert (final/'rng.pt').is_file() and load(final/'meta.json')['global_step']==270
        trace_sha=sha(root/'steps.jsonl')
        scope='Miles FSDP2 native DiT attention+MLP LoRA rank8;60s budget; pure raw SongBench Mean7; no guard/SP penalty.'
    assert len(trace)==270 and abs(trace[0]['kl'])<1e-8
    assert experiment['lr']==.0003 and experiment['beta']==.05
    assert (root/'completion.json').is_file()
    return dict(model='muse' if kind=='muse' else 'diffrhythm2',run=root.name,status='verified-complete',
        updates=270,experiment=experiment,experiment_sha256=sha(root/'experiment.json'),
        validation=validation,trace=trace,binding=stats,steps_sha256=trace_sha,scope=scope,
        kl_unit='sampled k3 / native completion token' if kind=='muse' else 'conditional Gaussian KL / active latent dimension',
        checkpoint=dict(step=270,files=[dict(name=str(p.relative_to(final)),bytes=p.stat().st_size,sha256=sha(p))
            for p in sorted(final.rglob('*')) if p.is_file()]))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--kind',choices=['muse','diff'],required=True)
    a=p.parse_args()
    try:record=collect(a.root,a.kind)
    except Exception as error:record=dict(status='rejected',run=a.root.name,error=f'{type(error).__name__}: {error}')
    print(json.dumps(dict(checked_at=datetime.now(timezone.utc).isoformat(),run=record),allow_nan=False))
