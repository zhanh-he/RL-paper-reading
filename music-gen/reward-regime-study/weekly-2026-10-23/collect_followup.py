"""One-shot manual recovery of job state and numeric training receipts; no monitor."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
PUBLIC=HERE.parents[2]/'platform/site/demos/weekly-2026-10-23'
remote=r'''
import hashlib,json,math,os
from pathlib import Path
root=Path('/storage/zhanh_storage/songbench-priority-20261020')
run=root/'outputs/muse-mean7-lr3e4-beta5e2-seed42-v3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def lines(p):
 if not p.exists():return []
 text=p.read_text();return [json.loads(x) for x in text.split('\n')[:-1] if x]
logs=list((run/'checkpoint').glob('*/logging.jsonl'))
rows=[r for r in lines(logs[0]) if 'global_step/max_steps' in r] if logs else []
trace=[dict(step=int(r['global_step/max_steps'].split('/')[0]),reward=r['reward'],kl=r['kl']) for r in rows]
assert [r['step'] for r in trace]==list(range(1,len(trace)+1))
scores=lines(run/'reward-bindings.jsonl')[:2*len(trace)]
assert len(scores)==2*len(trace)
paths={sha(p):p for p in (run/'decoded').glob('*.flac')}
invalid=0
for r in scores:
 assert r['audio_sha256'] in paths and r['scorer_identity']['mean7_only']
 if r.get('technical_invalid'):
  assert r['mean']==0 and all(v is None for v in r['songbench'].values());invalid+=1
 else:assert abs(sum(r['songbench'].values())/7-r['mean'])<1e-6
for i,r in enumerate(rows):
 assert abs(sum(s['mean'] for s in scores[2*i:2*i+2])/2-r['reward'])<1e-5
 assert r['learning_rate']==.0003 and math.isfinite(r['kl']) and math.isfinite(r['grad_norm'])
if trace:assert abs(trace[0]['kl'])<1e-8
identity=json.loads((run/'base-identity.json').read_text())
assert identity['model_sha256']=='c810b2a748579e06e960019f89a9cb98701e4ff545e2c760e8ceaa3343ff1b52'
alive=Path('/proc/4044350/cmdline').exists()
checkpoints=sorted(int(p.name.split('-')[1]) for p in (run/'checkpoint').glob('*/checkpoint-*'))
print(json.dumps(dict(alive=alive,trace=trace,step=len(trace),bound_audio_n=len(scores),technical_invalid_n=invalid,
 checkpoints=checkpoints,experiment=json.loads((run/'experiment.json').read_text()),
 source_sha256={p.name:sha(p) for p in (root/'code').iterdir() if p.suffix in ['.py','.sh','.pbs']},
 dependency_alive=Path('/proc/4044401/cmdline').exists(),
 scorer_identity=scores[0]['scorer_identity'] if scores else None)))
'''
lab=json.loads(subprocess.check_output(['ssh','lab5090','python3','-c',__import__('shlex').quote(remote)],text=True))
try:
    pbs=json.loads(subprocess.check_output(['ssh','gadi','qstat -f -F json 182083826.gadi-pbs 182084203.gadi-pbs'],text=True))
    states={key:value['job_state'] for key,value in pbs['Jobs'].items()}
except (subprocess.CalledProcessError,json.JSONDecodeError,KeyError):
    states={}
checked=datetime.now(timezone.utc).isoformat()
snapshot=dict(checked_at=checked,lab=lab,pbs_states=states,
  old_attempts=[dict(id='muse-mean7-v1',updates=0,status='failed-scorer-mode-mismatch'),
                dict(id='muse-mean7-v2',updates=33,status='failed-STFT40ms-before-first-checkpoint')],
  no_scheduled_tasks=True,no_other_jobs_interrupted=True)
(HERE/'followup-receipts.json').write_text(json.dumps(snapshot,indent=2,allow_nan=False)+'\n')
def gadi(job):
    state=states.get(job)
    return {'Q':'排队 · GPU资源不足','R':'运行中（结果待回收）','F':'已退出 · 待回收日志'}.get(state,'已提交 · 当前状态待读取')
jobs=[dict(name='Muse · Mean7 · LR3e-4',resource='5090 · PID4044350',
           status='训练中 · v3' if lab['alive'] else '已退出 · 待回收',progress=f'{lab["step"]}/270 更新；尚无完整验证'),
      dict(name='YuE2 · Mean7 · LR3e-4',resource='Gadi wa66 · 182083826',status=gadi('182083826.gadi-pbs'),progress='目标270更新 +4×30 valid；尚未回收更新'),
      dict(name='Muse · Mean7 · LR1e-4',resource='Gadi iv96 · 182084203',status=gadi('182084203.gadi-pbs'),progress='目标270更新 +4×30 valid；尚未回收更新'),
      dict(name='DiffRhythm2 · Mean7 · LR1e-4；Muse · SP重跑',resource='5090 · 有限依赖队列 PID4044401',
           status='等待前序训练完成' if lab['dependency_alive'] else '队列已退出 · 待回收',progress='各270更新，保存并评估0/50/100/270'),
      dict(name='Muse固定 Train / Test 评估',resource='5090 · 训练依赖链后执行',status='已接入依赖链 · 尚未执行',
           progress='每臂270 train +192×2 test；四个checkpoint；不做test选择'),
      dict(name='真实 YuE2-SongEval → SongBench 交叉评分',resource='5090 · 复用串行评分服务',status='已完整回收并补图',progress='120/120：0/50/100/270各30条'),
      dict(name='Kaya交叉评分重复任务189983',resource='Kaya ems011',status='已取消重复排队，转5090完成',progress='未消费GPU；其他用户暂停作业不变')]
traces=[dict(id='muse-mean7-live-v3',model='muse',reward='mean7',trace=lab['trace'],
             kl_unit='sampled k3 / native completion token',status=f'训练中 {lab["step"]}/270；不是完整验证结果')]
public=dict(checked_at=checked,protocol='SP/AI：CMI270 train /30 valid；WildSongBench192条件×2固定seed test。Lyrics2song原有CMI-test协议不改。',
            jobs=jobs,traces=traces,technical_invalid_policy='训练reward0，七维分数null；不以零分伪造SongBench测量。',
            receipt_sha256=hashlib.sha256((HERE/'followup-receipts.json').read_bytes()).hexdigest())
for folder in [HERE,PUBLIC]:(folder/'next-experiments.json').write_text(json.dumps(public,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(step=lab['step'],audio_bindings=lab['bound_audio_n'],checkpoints=lab['checkpoints'],pbs=states)))
