"""Manual one-shot collection, not a monitor or scheduler."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
JOBS = {
    '181297781.gadi-pbs': ('muse','richness'),
    '181297782.gadi-pbs': ('muse','instrument'),
    '181297783.gadi-pbs': ('muse','arrangement'),
    '181297497.gadi-pbs': ('heartmula','richness'),
    '181297498.gadi-pbs': ('heartmula','instrument'),
    '181297503.gadi-pbs': ('heartmula','arrangement'),
}


def remote(host, command):
    return subprocess.check_output(['ssh',host,command],text=True)


def main():
    jobs = json.loads(remote('gadi','qstat -f -F json '+' '.join(JOBS)))['Jobs']
    result = []
    fields = ['job_state','queue','project','Priority','Hold_Types','comment','Resource_List','ctime','Exit_status']
    for job, (model,reward) in JOBS.items():
        item = jobs[job]
        result.append(dict(job_id=job,model=model,reward=reward,lr=3e-4,beta=.05,scheduler='constant',
                           **{key:item[key] for key in fields if key in item}))
    diff = '/storage/zhanh_storage/backbone-rl-20261015/diff-formal'
    progress = json.loads(remote('lab5090',f'cat {diff}/diffrhythm2-richness-lr3e4-beta5e2-seed42/progress.json'))
    record = dict(snapshot_at=datetime.now(timezone.utc).isoformat(),no_scheduled_tasks=True,
        scope='Submission/current-phase evidence only; not completed formal results',jobs=result,
        diffrhythm2=dict(pid=3884225,process=remote('lab5090','ps -p 3884225 -o pid=,etime=,args=').strip(),
                        lr=3e-4,beta=.05,scheduler='constant',progress=progress),
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.iterdir()
                       if p.suffix in ['.py','.pbs','.sh','.patch']},
        superseded=dict(failed_before_training=['181226207','181226209'],
            unicode_parser='Physical JSONL parsing fixes285 splitlines fragments versus270 actual records; frozen SHA unchanged',
            cancelled_before_new_round=['181243929','181296053','181296054','181296055','181296998','181297001','181297003']))
    (ROOT/'launch-receipt.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(jobs=[(r['job_id'],r['job_state']) for r in result],diff=progress),indent=2))


if __name__ == '__main__':
    main()
