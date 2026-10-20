"""One-shot launch evidence; never schedules, polls or evaluates a model."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent


def remote(host, source):
    import shlex
    return json.loads(subprocess.check_output(['ssh',host,'python3 -c '+shlex.quote(source)],text=True))


def main():
    lab=remote('lab5090',r'''
from pathlib import Path
import ast,hashlib,json,re,subprocess
root=Path('/storage/zhanh_storage/songbench-priority-20261020')
accept=root/'acceptance/muse-richness-2steps-colocate4'
state_path=next((accept/'checkpoint').rglob('trainer_state.json'))
state=json.loads(state_path.read_text())
scores=[json.loads(line) for line in (accept/'reward-bindings.jsonl').open()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
audio_hashes={sha(p) for p in (accept/'decoded').rglob('*.flac')}
assert len(scores)==4 and all(s['audio_sha256'] in audio_hashes for s in scores)
assert state['global_step']==2 and state['log_history'][0]['kl']==0
assert all(r['grad_norm']>0 for r in state['log_history'])
checkpoint={p.name:sha(p) for p in state_path.parent.iterdir() if p.name in
 ['optimizer.pt','rng_state.pth','adapter_model.safetensors','adapter_config.json','trainer_state.json']}
assert {'optimizer.pt','rng_state.pth','adapter_model.safetensors'}<=checkpoint.keys()
formal=root/'outputs/muse-richness-lr3e4-beta5e2-seed42'
log=(formal/'train.log').read_text()
records=[ast.literal_eval(s) for s in re.findall(r"\{'loss':[^\r\n]+\}",log)]
assert records
print(json.dumps(dict(base=json.loads((formal/'base-identity.json').read_text()),
 acceptance=dict(steps=2,formal_training=False,trainable_parameters=5046272,
  scored_audio_bindings_verified=4,log_history=state['log_history'],checkpoint_sha256=checkpoint,
  scorer_identity=scores[0]['scorer_identity']),
 muse=dict(pid=3982658,status='running',latest=records[-1],experiment=json.loads((formal/'experiment.json').read_text())),
 diff=dict(pid=3983215,status='finite-dependency-queue',predecessor='Muse-SP full training and validation',
  process=subprocess.check_output(['ps','-p','3983215','-o','pid=,args='],text=True).strip()))))
''')
    rewards={'181718185.gadi-pbs':'instrument','181718187.gadi-pbs':'arrangement','181718819.gadi-pbs':'mean7'}
    raw=subprocess.check_output(['ssh','gadi','qstat -f -F json '+' '.join(rewards)],text=True)
    jobs=json.loads(raw)['Jobs']
    record=dict(checked_at=datetime.now(timezone.utc).isoformat(),no_scheduled_tasks=True,
        source_sha256={str(p.relative_to(ROOT.parent)):hashlib.sha256(p.read_bytes()).hexdigest()
          for p in list(ROOT.glob('*.py'))+list(ROOT.glob('*.pbs'))+list(ROOT.glob('*.sh'))+
          [ROOT.parent/n for n in ['rewards.py','reward_server.py','muse_plugin.py','mucodec_fixed_seed.py','runtime_entry.py','validate_muse.py','check_muse_base.py']]},
        **lab,gadi=[dict(job_id=key,model='YuE2' if rewards[key]=='mean7' else 'Muse',reward=rewards[key],
            **{field:value[field] for field in ['job_state','queue','project','Priority','Resource_List','comment'] if field in value})
            for key,value in jobs.items()],superseded_held_jobs_deleted=['181714731','181714736','181714740'],
        scope='Launch snapshot only. No completed formal Muse or Mean7 result; validation/test not yet available.',
        yue2_gate=dict(cpu_native_tokenizer_import='passed on5090 and Gadi',gpu_native_grpo='not yet accepted',
            queue='2 native updates -> separate process resume2to3 -> fresh270 formal; any failed gate stops',
            max_semantic_tokens=1500,seconds_budget=60,framework='TRL GRPOTrainer with project native rollout/likelihood adapter'),
        duration_note='Muse old SP/subhead controls use512 native completion tokens and10.24s fixed-seed codec, not the new Mean7 60s protocol. Cross-host baseline audio will be rechecked on recovery; do not imply bit-identical Gadi/5090 generation.',
        failures_repaired=['stdlib container entry replaces missing generate_objective dependency',
          'single GPU uses colocated vLLM, not invalid same-device actor/server NCCL',
          'bounded1GiB KV cache and codec/actor allocator-cache release prevent scorer/decode OOM'])
    destination=ROOT.parent/'runs/2026-10-20-songbench-priority/launch.json'
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(muse=record['muse']['latest']['global_step/max_steps'],
        jobs=[(j['job_id'],j['job_state']) for j in record['gadi']],diff=record['diff']['status'])))


if __name__=='__main__':main()
