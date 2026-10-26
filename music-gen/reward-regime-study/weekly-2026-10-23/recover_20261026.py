"""One-shot CPU recovery; never starts jobs or converts incomplete scores to means."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def lab():
    from collect_new import collect, load, sha, verify, HEADS
    root = Path('/storage/zhanh_storage/songbench-priority-20261020/outputs')
    definitions = [
        ('muse-mean7-lr3e4-beta5e2-seed42-v3', 'muse', .0003),
        ('muse-richness-lr3e4-beta5e2-seed42-v2', 'muse', .0003),
        ('diff-mean7-lr1e4-formal-20261023/diffrhythm2-mean7-lr1e4-beta5e2-seed42', 'diff', .0001),
    ]
    runs = [collect(root / name, kind, lr, [0, 50, 100, 270]) for name, kind, lr in definitions]
    cohorts = []
    run = root / definitions[0][0]
    for step in [0, 50, 100, 270]:
        receipt_path = run / 'full-cohorts' / f'train_{step:06d}' / 'receipt.json'
        receipt = load(receipt_path)
        rows = receipt['rows']
        assert receipt['n'] == len(rows) == 270 and [r['index'] for r in rows] == list(range(270))
        folder = root / 'shared-muse-baseline-train' if step == 0 else receipt_path.parent
        stats = {'bound_audio_n': 0}
        measured = []
        cleaned = []
        for r in rows:
            meta = load(folder / f'{r["index"]:03d}' / 'generation.json')
            assert meta['split'] == 'train' and meta['seed'] == 6101 + r['index']
            assert meta['data_sha256'] == 'ff437d751d62742d92daf7ac56a678010c52d6c22def5624a8324301e6699be6'
            if r.get('audio_sha256'):
                assert meta['audio_sha256'] == r['audio_sha256']
                verify(r, folder / f'{r["index"]:03d}' / 'audio.flac', 'mean7', stats)
            else:
                assert meta.get('invalid') and r['mean'] == 0
            heads = r.get('songbench')
            value = sum(heads[h] for h in HEADS) / 7 if heads and all(heads.get(h) is not None for h in HEADS) else None
            if value is not None:
                assert math.isfinite(value)
                measured.append(value)
            cleaned.append(dict(index=r['index'], audio_sha256=r.get('audio_sha256'), mean7=value,
                                songbench=heads, technical_invalid=value is None))
        mean = sum(measured) / 270 if len(measured) == 270 else None
        assert mean == receipt['SongBench_mean7'] or (mean is not None and abs(mean-receipt['SongBench_mean7']) < 1e-12)
        assert len(measured) == receipt['songbench_scored_n']
        cohorts.append(dict(model='muse', reward='mean7', lr=.0003, split='train', step=step,
                            n=270, scored_n=len(measured), mean7=mean, invalid_n=270-len(measured),
                            status='complete' if mean is not None else 'complete-with-unscorable-outputs',
                            receipt_sha256=sha(receipt_path), binding=stats, rows=cleaned))
    test_baseline = root / 'shared-muse-baseline-test'
    generation = [load(p) for p in sorted(test_baseline.glob('*/generation.json'))]
    test_log = run / 'full-cohorts/test.log'
    assert 'No complete Mean7 mean for invalid technical audio: no AUDIO tokens' in test_log.read_text()
    return dict(runs=runs, cohorts=cohorts, test=dict(status='failed-evaluation', generated=len(generation),
        target=384, missing_audio_n=sum(r.get('audio_sha256') is None for r in generation),
        completed_receipts=len(list((run/'full-cohorts').glob('test_*/receipt.json'))),
        reason='No AUDIO tokens in frozen baseline; evaluator raised instead of recording unavailable heads.',
        log_sha256=sha(test_log)), auditor_sha256=sha(Path(__file__)))


def gadi():
    from collect_new import load, sha, verify, BASE_SHA
    root = Path('/scratch/wa66/hm6416/songbench-priority-20261020/outputs')
    muse = root / 'muse-mean7-lr1e4-beta5e2-seed42'
    final, = list((muse/'checkpoint').glob('*/checkpoint-270'))
    state = load(final/'trainer_state.json')
    history = [r for r in state['log_history'] if 'reward' in r]
    assert state['global_step'] == 270 and [r['step'] for r in history] == list(range(1,271))
    scores = [json.loads(line) for line in (muse/'reward-bindings.jsonl').read_text().splitlines()]
    assert len(scores) == 540
    paths = {sha(p):p for p in (muse/'decoded').glob('*.flac')}
    binding = {'bound_audio_n':0}
    for score in scores:
        verify(score, paths[score['audio_sha256']], 'mean7', binding)
    for i,row in enumerate(history):
        assert row['learning_rate'] == .0001 and math.isfinite(row['kl']) and math.isfinite(row['grad_norm'])
        assert abs(sum(r['mean'] for r in scores[2*i:2*i+2])/2-row['reward']) < 1e-6
    assert load(muse/'base-identity.json')['model_sha256'] == BASE_SHA
    validation_log = muse/'valid.log'
    if not validation_log.exists():
        validation_log, = [p for p in muse.glob('*.log') if 'GCC 9 or later' in p.read_text(errors='replace')]
    assert 'GCC 9 or later' in validation_log.read_text(errors='replace')
    yue = root/'yue2-mean7-lr3e4-beta5e2-seed42-v2/acceptance.log'
    assert "object has no attribute 'rotary_emb'" in yue.read_text(errors='replace')
    trace = [dict(step=r['step'], reward=r['reward'], kl=r['kl']) for r in history]
    return dict(muse=dict(id='muse-mean7-lr1e4-training-only', model='muse', reward='mean7', lr=.0001,
        status='270步训练完成 · 验证环境失败；不是完整验证结果', phase='训练完成 / 验证失败',
        trace=trace, binding=binding, kl_unit='sampled k3 / native completion token',
        checkpoint=dict(step=270, files=[dict(name=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(final.iterdir()) if p.is_file()]),
        validation_log_sha256=sha(validation_log), steps_sha256=sha(final/'trainer_state.json')),
        yue2=dict(status='failed-acceptance-before-update',updates=0,reason="PEFT/native NAR path: YuE2ForCausalLM has no rotary_emb",log_sha256=sha(yue)))


def run():
    results = {}
    for host, python in [('lab5090','python3'),('gadi','python3')]:
        directory = '/tmp/rl-recovery-20261026' if host=='lab5090' else '/scratch/wa66/hm6416/songbench-priority-20261020/recovery-20261026'
        subprocess.run(['ssh',host,'mkdir','-p',directory],check=True)
        subprocess.run(['scp',str(HERE/'collect_new.py'),str(Path(__file__)),f'{host}:{directory}/'],check=True)
        result = subprocess.check_output(['ssh',host,python,f'{directory}/recover_20261026.py','--'+('lab' if host=='lab5090' else 'gadi')],text=True)
        results[host] = json.loads(result)
    queue = subprocess.check_output(['ssh','kaya','squeue -h -u "$(whoami)" -o "%i|%j|%T|%r"'],text=True)
    jobs = [dict(zip(['id','name','state','reason'],line.split('|'))) for line in queue.splitlines()]
    results['kaya'] = dict(jobs=jobs, held_n=sum(j['reason']=='JobHeldUser' for j in jobs))
    record = dict(checked_at=datetime.now(timezone.utc).isoformat(),machines=results,
                  policy='Read-only CPU/hash recovery. No jobs started, stopped, released or scheduled.')
    (HERE/'recovery-2026-10-26.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(verified_runs=[r['run'] for r in results['lab5090']['runs']],cohorts=4,
                         gadi_muse_steps=len(results['gadi']['muse']['trace']),kaya_held=results['kaya']['held_n'])))


if __name__=='__main__':
    if '--lab' in sys.argv:print(json.dumps(lab(),allow_nan=False))
    elif '--gadi' in sys.argv:print(json.dumps(gadi(),allow_nan=False))
    else:run()
