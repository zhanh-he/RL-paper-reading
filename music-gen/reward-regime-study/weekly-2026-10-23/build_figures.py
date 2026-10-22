"""Reproducible paired analysis and twelve presentation-ready figure exports."""
from datetime import datetime,timezone
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
PUBLIC=ROOT/'platform/site/demos/weekly-2026-10-23'
HEADS=['Vocal','Instrumental','Melody','Structure','Arrangement','Mixing','Musicality']
COLORS={'richness':'#14816d','instrument':'#3466a6','arrangement':'#b07822','mean7':'#a8446c','songeval':'#3466a6'}
REWARDS={'richness':'SP Richness','instrument':'SB Instrument','arrangement':'SB Arrangement','mean7':'SB Mean7'}
MODEL={'yue2':'YuE2','diffrhythm2':'DiffRhythm2','muse':'Muse'}
PUBLIC.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':15,'axes.titlesize':19,
    'axes.labelsize':16,'xtick.labelsize':13,'ytick.labelsize':14,'svg.fonttype':'none',
    'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#89918f',
    'text.color':'#202a28','axes.labelcolor':'#202a28','xtick.color':'#48514e','ytick.color':'#48514e',
    'savefig.facecolor':'white','figure.facecolor':'white'})


def read(name):return json.loads((HERE/name).read_text())


def stats(a,b):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    assert len(a)==len(b)==30 and np.isfinite(a).all() and np.isfinite(b).all()
    delta=b-a
    rng=np.random.default_rng(20261023)
    bootstrap=delta[rng.integers(0,30,size=(10000,30))].mean(1)
    lo,hi=np.quantile(bootstrap,[.025,.975])
    return dict(n=30,baseline=float(a.mean()),final=float(b.mean()),delta=float(delta.mean()),
        ci95=[float(lo),float(hi)],positive=int((delta>0).sum()),paired_deltas=delta.tolist())


legacy=read('legacy-rechecked.json')
assert len(legacy['runs'])==6 and all(r['status']=='verified-historical-comparator' for r in legacy['runs'])
new=[read('muse-sp.json')['run'],read('diff-mean7.json')['run']]
assert all(r['status']=='verified-complete' for r in new)
raw=legacy['runs']+new
runs=[]
for r in raw:
    before=r['validation'][0];after=r['validation'][-1]
    reward=r['experiment']['reward']
    record=dict(id=r['run'],model=r['model'],reward=reward,updates=r['updates'],
        experiment=r['experiment'],scope=r.get('scope','Historical guarded SP/subhead control; not matched across architectures'),
        binding=r['binding'],checkpoint=r['checkpoint'],experiment_sha256=r['experiment_sha256'],
        steps_sha256=r['steps_sha256'],trace=r['trace'],kl_unit=r['kl_unit'],validation=r['validation'],
        target=stats([s['target'] for s in before['rows']],[s['target'] for s in after['rows']]),
        mean7=stats([s['mean7'] for s in before['rows']],[s['mean7'] for s in after['rows']]),
        heads={k:stats([s['songbench'][k] for s in before['rows']],[s['songbench'][k] for s in after['rows']]) for k in HEADS})
    record['sp']=stats([s['sp'] for s in before['rows']],[s['sp'] for s in after['rows']]) if before['means']['SP_Richness'] is not None else None
    record['median_seconds']=float(np.median([s['seconds'] for s in after['rows']]))
    runs.append(record)
songeval=read('songeval.json')['runs']
for r in songeval:
    r['effect']=stats([s['value'] for s in r['validation']['0']['rows']],[s['value'] for s in r['validation']['270']['rows']])
assert len(songeval)==3
# Container bytes are independently bound; common baselines require identical decoded PCM.
assert all([(x['pcm_sha256'],x['sample_rate'],x['frames']) for x in r['validation']['0']['rows']]==
           [(x['pcm_sha256'],x['sample_rate'],x['frames']) for x in songeval[0]['validation']['0']['rows']] for r in songeval)
diff_baselines=[[x['audio_sha256'] for x in r['validation'][0]['rows']] for r in runs if r['model']=='diffrhythm2']
assert all(x==diff_baselines[0] for x in diff_baselines)


def row(label,effect,color,short=None):return dict(label=label,short=short or label,effect=effect,color=color)


definitions=[dict(id='01-songeval',title='YuE2: SongEval validation after GRPO',
    subtitle='CMI-Pref · 270 train / 30 fixed validation conditions · endpoint step 270',
    panels=[dict(title='SongEval mean · native score units',unit='SongEval score',max=5,
        rows=[row('LR '+f'{r["lr"]:.0e}'.replace('e-0','e-'),r['effect'],COLORS['songeval']) for r in songeval])],
    note='Beta 0.01 · rank-4 q/v LoRA · 600 semantic-token cap. LR selection and intervals are exploratory.'),
 dict(id='02-sp-transfer',title='Does SP richness transfer to learned music scores?',
    subtitle='Same three SP-trained arms · raw SP and SongBench use different units',
    panels=[dict(title='Raw SP richness · proxy units',unit='SP richness',max=1,
        rows=[row(MODEL[r['model']],r['sp'],COLORS['richness']) for r in runs if r['reward']=='richness']),
        dict(title='SongBench mean7 · cross-score',unit='SongBench mean7',max=10,
        rows=[row(MODEL[r['model']],r['mean7'],COLORS['instrument']) for r in runs if r['reward']=='richness'])],
    note='Different model budgets / adapters / beta: within-arm effects only, not a causal comparison of model quality.'),
 dict(id='03-reward-regimes',title='Reward choice: paired change in SongBench mean7',
    subtitle='All eight verified arms · same fixed 30-condition validation membership',
    panels=[dict(title='SongBench mean7 · native score units',unit='SongBench mean7',max=10,
        rows=[row(MODEL[r['model']]+' · '+REWARDS[r['reward']],r['mean7'],COLORS[r['reward']],
            MODEL[r['model']]+'\n'+REWARDS[r['reward']]) for r in runs])],
    note='SP / single-head targets include guard; Mean7 is unguarded. Mean7 is not a human MOS or independent judge.'),
 dict(id='04-diff-mean7',title='DiffRhythm2: what changed under Mean7-only GRPO?',
    subtitle='Pure seven-head mean objective · LR 3e-4 / beta 0.05 / rank 8 · native 60s budget',
    panels=[dict(title='Seven SongBench heads · native score units',unit='SongBench score',max=10,
        rows=[row(k,next(r for r in runs if r['reward']=='mean7')['heads'][k],COLORS['mean7']) for k in HEADS])],
    note='All 270 updates and 0 / 50 / 100 / 270 validation verified. Fixed final endpoint; no best-step selection.')]


def clean(ax,orientation):
    ax.set_axisbelow(True)
    ax.grid(axis='x' if orientation=='h' else 'y',color='#e4e9e7',lw=.8)
    ax.tick_params(length=0,pad=9)


def draw(definition,variant):
    panels=definition['panels'];fig,axes=plt.subplots(1,len(panels),figsize=(16,9),squeeze=False)
    fig.subplots_adjust(left=.22 if variant in ['A','C'] and len(panels)==1 else .12,
        right=.95,top=.78,bottom=.28 if variant=='C' else .23 if variant=='B' else .19,wspace=.43)
    fig.text(.065,.925,definition['title'],fontsize=26,fontweight='bold',ha='left')
    fig.text(.065,.872,definition['subtitle'],fontsize=14,color='#56615b')
    fig.text(.95,.925,variant,fontsize=24,fontweight='bold',color='#56615b',ha='right')
    for ax,panel in zip(axes.flat,panels):
        rows=panel['rows'];n=len(rows);effects=[r['effect'] for r in rows]
        values=np.array([r['delta'] for r in effects]);bounds=np.array([r['ci95'] for r in effects])
        colors=[r['color'] for r in rows]
        ax.set_title(panel['title'],loc='left',pad=20,fontweight='bold')
        if variant=='A':
            y=np.arange(n);ax.barh(y,values,color=colors,height=.56,zorder=2)
            ax.errorbar(values,y,xerr=[values-bounds[:,0],bounds[:,1]-values],fmt='none',ecolor='#202a28',capsize=4,lw=1.3,zorder=3)
            ax.axvline(0,color='#48514e',lw=1)
            ax.set_yticks(y,[r['label'] for r in rows]);ax.invert_yaxis();ax.set_xlabel('Change vs baseline · '+panel['unit'])
            lo=min(0,float(bounds[:,0].min()));hi=max(0,float(bounds[:,1].max()));span=max(hi-lo,.01)
            ax.set_xlim(lo-.12*span,hi+.27*span)
            for i,e in enumerate(effects):ax.text(hi+.055*span,i,f'{e["delta"]:+.3f}',va='center',fontsize=14,fontweight='bold')
            clean(ax,'h')
        elif variant=='B':
            x=np.arange(n);ax.bar(x,values,color=colors,width=.62,zorder=2)
            ax.errorbar(x,values,yerr=[values-bounds[:,0],bounds[:,1]-values],fmt='none',ecolor='#202a28',capsize=5,lw=1.3,zorder=3)
            ax.axhline(0,color='#48514e',lw=1)
            ax.set_xticks(x,[r['short'] for r in rows]);ax.set_ylabel('Change vs baseline · '+panel['unit'])
            if n>5:ax.tick_params(axis='x',labelsize=11)
            lo=min(0,float(bounds[:,0].min()));hi=max(0,float(bounds[:,1].max()));span=max(hi-lo,.01)
            ax.set_ylim(lo-.14*span,hi+.25*span)
            for i,e in enumerate(effects):ax.text(i,bounds[i,1]+.055*span,f'{e["delta"]:+.3f}',ha='center',fontsize=13,fontweight='bold')
            clean(ax,'v')
        else:
            # Numeric before/after segments, with exact paired-change intervals alongside.
            y=np.arange(n)
            for i,(r,e) in enumerate(zip(rows,effects)):
                ax.plot([e['baseline'],e['final']],[i,i],color=r['color'],lw=3)
                ax.scatter(e['baseline'],i,facecolor='white',edgecolor='#606a65',s=75,zorder=3,linewidth=1.4)
                ax.scatter(e['final'],i,color=r['color'],s=75,zorder=4,edgecolor='white',linewidth=.7)
                ax.text(1.02,i,f'{e["baseline"]:.3f} → {e["final"]:.3f}\nΔ {e["delta"]:+.3f} [{e["ci95"][0]:+.3f}, {e["ci95"][1]:+.3f}]',
                    transform=ax.get_yaxis_transform(),va='center',fontsize=11 if n>5 else 13)
            lo=min(e['baseline'] for e in effects);hi=max(e['final'] for e in effects)
            lo=min(lo,min(e['final'] for e in effects));hi=max(hi,max(e['baseline'] for e in effects))
            span=max(hi-lo,.02)
            ax.set_yticks(y,[r['label'] for r in rows]);ax.invert_yaxis();ax.set_xlim(max(0,lo-.15*span),min(panel['max'],hi+.15*span))
            ax.set_xlabel(panel['unit']+' · focused axis\nBaseline ○ / step270 ●');clean(ax,'h')
            pos=ax.get_position();ax.set_position([pos.x0,pos.y0,pos.width*.65,pos.height])
    fig.text(.065,.095,'N=30 paired conditions · 95% condition-bootstrap CI · 10,000 resamples · one training seed',fontsize=12,color='#56615b')
    fig.text(.065,.065,definition['note'],fontsize=11.2,color='#56615b')
    fig.text(.065,.036,'Validation, not full-song test or human preference. Unadjusted intervals; no across-model causal ranking.',fontsize=11.2,color='#56615b')
    files={}
    for suffix in ['png','pdf','svg']:
        name=f'{definition["id"]}-{variant}.{suffix}';path=PUBLIC/name
        fig.savefig(path,dpi=150 if suffix=='png' else None)
        files[suffix]=dict(file=name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
    plt.close(fig);return files


figures=[]
for definition in definitions:
    figures.append(dict(id=definition['id'],title=definition['title'],subtitle=definition['subtitle'],
        panels=definition['panels'],note=definition['note'],variants={v:draw(definition,v) for v in 'ABC'}))
summary=dict(updated_at=datetime.now(timezone.utc).isoformat(),run_count=len(runs),runs=runs,songeval=songeval,
    audited_audio_bindings=sum(r['binding']['bound_audio_n'] for r in runs),
    method=dict(n=30,endpoint=270,bootstrap_draws=10000,bootstrap_seed=20261023,
        ci='Paired condition-percentile bootstrap95%; not across-training-seed uncertainty; unadjusted multiple comparisons',
        scope='Frozen CMI270 train /30 validation. No full-test or human MOS claim. Short YuE2/Muse versus60s Diff budgets disclosed.'),
    failures=[dict(model='Muse',reward='SongBench-Instrument',job='181718185',status='failed-before-training',
        reason='libpython3.11.so.1.0 missing at scorer bootstrap; no train.log or optimizer updates'),
        dict(model='Muse',reward='SongBench-Arrangement',job='181718187',status='failed-before-training',
        reason='libpython3.11.so.1.0 missing at scorer bootstrap; no train.log or optimizer updates'),
        dict(model='YuE2',reward='SongBench-Mean7 / TRL',job='181718819',status='failed-acceptance-before-update',
        reason='YuE2ForCausalLM gradient-checkpointing integration rejected by Transformers during TRL init; no formal run')],
    figures=figures)
(HERE/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
(PUBLIC/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
with (PUBLIC/'endpoint-results.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['model','objective','metric','n','baseline','step270','delta','ci95_low','ci95_high','positive_pairs'])
    for r in runs:
        for metric,s in [('SongBench_mean7',r['mean7']),('target_reward',r['target']),('SP_Richness',r['sp'])]:
            if s:w.writerow([r['model'],r['reward'],metric,30,s['baseline'],s['final'],s['delta'],*s['ci95'],s['positive']])
    for r in songeval:
        s=r['effect'];w.writerow(['YuE2','SongEval '+str(r['lr']),'SongEval_mean5',30,s['baseline'],s['final'],s['delta'],*s['ci95'],s['positive']])
with zipfile.ZipFile(PUBLIC/'figures-ABC.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(PUBLIC.iterdir()):
        if p.suffix in ['.png','.pdf','.svg','.csv','.json']:z.write(p,p.name)
print(json.dumps(dict(figures=12,formats=['PNG','PDF','SVG'],runs=len(runs),bindings=summary['audited_audio_bindings'],
    effects=[(r['model'],r['reward'],r['mean7']['delta'],r['mean7']['ci95']) for r in runs]),indent=2))
