"""Render measured SongBench progression; absent splits/arms remain absent."""
from pathlib import Path
import csv
import hashlib
import json
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PUBLIC = ROOT / 'platform/site/demos/weekly-2026-10-23'
SOURCE = HERE / 'summary.json'
data = json.loads(SOURCE.read_text())
STEPS = [0, 50, 100, 270]
MODELS = {'diffrhythm2': 'DiffRhythm2', 'yue2': 'YuE2', 'muse': 'Muse'}
ARMS = {
    'richness': ('SP Richness-GRPO', '#16836f', 'o'),
    'mean7': ('SongBench Mean7-GRPO', '#ad426b', 'D'),
    'instrument': ('SongBench Instrument-GRPO', '#346caf', 's'),
    'arrangement': ('SongBench Arrangement-GRPO', '#b57919', '^'),
}
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 16, 'axes.labelsize': 17,
    'axes.titlesize': 28, 'xtick.labelsize': 15, 'ytick.labelsize': 15,
    'pdf.fonttype': 42, 'svg.fonttype': 'none', 'savefig.facecolor': 'auto',
})


def checkpoint(run, step):
    receipts = [v for v in run['validation'] if v['step'] == step]
    assert len(receipts) <= 1
    if not receipts:
        return None
    v = receipts[0]
    assert v['n'] == len(v['rows']) == 30
    assert len({r['index'] for r in v['rows']}) == 30
    heads = ['Vocal', 'Instrumental', 'Melody', 'Structure', 'Arrangement', 'Mixing', 'Musicality']
    # Independently recompute the plotted value from all seven per-audio heads.
    means = [sum(row['songbench'][h] for h in heads) / 7 for row in v['rows']]
    value = float(np.mean(means))
    assert abs(value - v['means']['SongBench_mean7']) < 1e-12
    return dict(step=step, value=value, n=30, receipt_sha256=v['receipt_sha256'])


def draw(model, series, variant):
    dark = variant == 'B'
    bg = '#151b1a' if dark else '#ffffff'
    fg = '#eef3f0' if dark else '#202d28'
    muted = '#afbeb6' if dark else '#617168'
    grid = '#35433c' if dark else '#e1e7e3'
    baseline = series[0]['points'][0]['value']
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=bg)
    ax.set_facecolor(bg)
    fig.subplots_adjust(left=.10, right=.75, top=.75, bottom=.24)
    fig.text(.10, .925, MODELS[model], fontsize=32, fontweight='bold', color=fg)
    fig.text(.10, .865, 'SongBench mean · steps 0 / 50 / 100 / 270', fontsize=21, color=fg)
    fig.text(.10, .808, 'Fixed validation · N = 30 · LR = 3e-4 · one training seed', fontsize=15, color=muted)
    ax.set_xlim(-8, 282)
    ax.set_xticks(STEPS)
    ax.set_xlabel('GRPO update step', labelpad=14, color=fg)
    ax.set_ylabel('SongBench seven-head mean (0–10)', labelpad=12, color=fg)
    vals = [p['value'] for s in series for p in s['points'] if p]
    if variant == 'C':
        span = max(max(vals)-min(vals), .12)
        ax.set_ylim(min(vals)-span*.25, max(vals)+span*.30)
        fig.text(.75, .865, 'FOCUSED Y AXIS', ha='right', fontsize=12, color=muted)
    else:
        ax.set_ylim(3.7, 6.3)
        ax.yaxis.set_major_locator(MultipleLocator(.5))
        fig.text(.75, .865, 'SAME Y SCALE · ALL MODELS', ha='right', fontsize=11, color=muted)
    ax.grid(axis='y', color=grid, lw=.9, zorder=0)
    ax.tick_params(colors=muted, length=0, pad=10)
    for side, spine in ax.spines.items():
        spine.set_visible(side in ['bottom', 'left'])
        spine.set_color(grid)
    ax.axhline(baseline, color='#b8c4bd' if dark else '#69756e',
               lw=1.6, ls=(0, (5, 4)), zorder=1)
    colors = {'richness': '#51c3aa', 'mean7': '#f284ac', 'instrument': '#80b5f4',
              'arrangement': '#e8b85a'} if dark else {k: v[1] for k, v in ARMS.items()}
    for s in series:
        available = [p for p in s['points'] if p]
        color = colors[s['reward']]
        ax.plot([p['step'] for p in available], [p['value'] for p in available],
                color=color, marker=ARMS[s['reward']][2], ms=9, lw=2.7,
                markeredgecolor=bg, markeredgewidth=1.2, label=s['label'], zorder=3)
    # A side key keeps endpoint labels from colliding when scores are close.
    for i, s in enumerate(series):
        final = s['points'][-1]['value']
        color = colors[s['reward']]
        y = .72 - i*.13
        fig.text(.79, y, s['label'].replace('-GRPO', ''), fontsize=14, color=color, fontweight='bold')
        fig.text(.79, y-.037, f'Valid · 270: {final:.3f}  ({final-baseline:+.3f})', fontsize=13, color=fg)
    fig.text(.79, .17, f'Baseline · step 0\n{baseline:.3f}', fontsize=15, color=muted, linespacing=1.5)
    fig.lines.append(plt.Line2D([.79, .835], [.235, .235], transform=fig.transFigure,
                              color=muted, lw=1.6, ls=(0, (5, 4))))
    fig.text(.10, .138, 'Measured validation only. Fixed-train and full-test checkpoint scores are unavailable.',
             fontsize=13, color=muted)
    missing_note = ('Muse: steps 50 / 100 not evaluated; the line connects measured endpoints only.' if model == 'muse'
                    else 'Markers are measured checkpoints; lines only connect them. No smoothing or filled-in scores.')
    fig.text(.10, .096, missing_note, fontsize=13, color=muted)
    fig.text(.10, .054, 'Objectives retain their recorded training names. Exact scores and receipt hashes accompany the figures.',
             fontsize=12, color=muted)
    files = {}
    for suffix in ['png', 'pdf', 'svg']:
        name = f'progression-{model}-{variant}.{suffix}'
        path = PUBLIC / name
        fig.savefig(path, dpi=150 if suffix == 'png' else None)
        if suffix == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
        files[suffix] = dict(file=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)
    plt.close(fig)
    return files


models = []
rows = []
for model, name in MODELS.items():
    runs = sorted([r for r in data['runs'] if r['model'] == model], key=lambda r: list(ARMS).index(r['reward']))
    series = []
    for r in runs:
        assert r['experiment']['lr'] == .0003
        points = [checkpoint(r, step) for step in STEPS]
        assert points[0] and points[-1]
        s = dict(id=r['id'], model=model, reward=r['reward'], label=ARMS[r['reward']][0],
                 split='valid', lr=r['experiment']['lr'], points=points, steps_sha256=r['steps_sha256'])
        series.append(s)
        for split in ['train', 'valid', 'test']:
            for step, p in zip(STEPS, points):
                measured = split == 'valid' and p is not None
                rows.append([model, r['id'], s['label'], r['experiment']['lr'], split, step,
                             p['value'] if measured else '', 30 if measured else '',
                             'measured' if measured else 'unavailable', p['receipt_sha256'] if measured else ''])
    assert all(abs(s['points'][0]['value'] - series[0]['points'][0]['value']) < 1e-12 for s in series)
    models.append(dict(id=model, title=name, series=series, baseline=series[0]['points'][0]['value'],
                       variants={v: draw(model, series, v) for v in 'ABC'}))

manifest = dict(metric='SongBench_mean7', metric_scale=[0, 10], steps=STEPS,
                source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                source_updated_at=data['updated_at'], source_file='summary.json',
                split='valid', n=30, lr=.0003, models=models,
                unavailable=['fixed train checkpoint evaluation', 'full test checkpoint evaluation',
                             'small-learning-rate SongBench arm', 'SongEval arms scored with SongBench',
                             'Muse steps 50 and 100', 'Muse SongBench reward arms'],
                variants={'A': 'Unified scale', 'B': 'Dark presentation / unified scale', 'C': 'Focused scale'},
                missing_policy='No copied split scores, no interpolation of missing checkpoints, no reward relabeling. Lines only connect measured markers.')
for directory in [HERE, PUBLIC]:
    (directory / 'progression.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    with (directory / 'progression.csv').open('w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['model', 'run_id', 'reward', 'lr', 'split', 'step', 'SongBench_mean7', 'n', 'status', 'receipt_sha256'])
        w.writerows(rows)
with zipfile.ZipFile(PUBLIC / 'progression-ABC.zip', 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for m in models:
        for v in m['variants'].values():
            for file in v.values():
                z.write(PUBLIC / file['file'], file['file'])
    for name in ['progression.csv', 'progression.json']:
        z.write(PUBLIC / name, name)
print(json.dumps(dict(models=3, variants=9, measured_series=sum(len(m['series']) for m in models),
                      measured_points=sum(p is not None for m in models for s in m['series'] for p in s['points'])), indent=2))
