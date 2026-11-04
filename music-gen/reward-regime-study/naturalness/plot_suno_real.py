"""Exploratory source-backed plots; no human defect labels or independent48 n."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];DATA=HERE/'runs/2026-11-04-suno-real/audit.json';OUT=ROOT/'platform/site/demos/visuals'
def main():
 d=json.loads(DATA.read_text());rows=d['samples'];metrics=[('modulation_depth','Auditory modulation depth'),('roughness70','DW roughness at assumed 70 dB [asper]'),('resonance_protected','Narrow-peak proxy, harmonic veto')]
 plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(1,3,figsize=(12.5,4.2),layout='constrained')
 for ax,(key,title) in zip(axes,metrics):
  x=[r['signal']['rms_dbfs'] for r in rows];y=[r['psycho'][key] for r in rows]
  for lang,color in [('cn','#147d78'),('en','#b96832')]:
   ii=[i for i,r in enumerate(rows) if '_'+lang+'_' in r['id']];ax.scatter(np.array(x)[ii],np.array(y)[ii],s=26,alpha=.75,label=lang.upper(),color=color)
  ax.set(xlabel='Original excerpt RMS [dBFS]',ylabel=title);ax.grid(alpha=.15);ax.legend(frameon=False)
 fig.suptitle('Real Suno V5 / Muse release: 16 songs, 8 source families, 48 unmodified 10 s excerpts\nDescriptors versus original level; no human artifact labels; points are clustered within songs',fontsize=11)
 for ext in ['png','svg']:fig.savefig(OUT/('suno-real-distributions.'+ext),dpi=180)
 plt.close(fig)
 families=sorted({r['source_family'] for r in rows});fig,axes=plt.subplots(1,3,figsize=(12.5,4.5),layout='constrained')
 for ax,(key,title) in zip(axes,metrics):
  for index,family in enumerate(families):
   rr=sorted([r for r in rows if r['source_family']==family and r['source_fraction']==.5],key=lambda r:r['track_index'])
   y=[r['psycho'][key] for r in rr];ax.plot([index-.1,index+.1],y,'-',color='#a0a0a0',linewidth=1);ax.scatter([index-.1],[y[0]],color='#147d78',s=24);ax.scatter([index+.1],[y[1]],color='#b96832',s=24)
  ax.set_xticks(range(len(families)),[f.replace('suno_','') for f in families],rotation=55,ha='right');ax.set_ylabel(title);ax.grid(axis='y',alpha=.15)
 axes[0].scatter([],[],color='#147d78',label='track 0');axes[0].scatter([],[],color='#b96832',label='track 1');axes[0].legend(frameon=False)
 fig.suptitle('Fixed midpoint excerpts from real Suno source families\nConnections mean shared family ID, not verified identical generation input or phrase alignment',fontsize=11)
 for ext in ['png','svg']:fig.savefig(OUT/('suno-real-family-p50.'+ext),dpi=180)
 plt.close(fig)
 for path in OUT.glob('suno-real-*.svg'):
  path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
if __name__=='__main__':main()
