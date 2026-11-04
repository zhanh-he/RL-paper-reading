"""Scientific response plots from the published measured psychoacoustic run."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
run=HERE/'runs/2026-11-04-psychoacoustic'
data=json.loads((run/'audit.json').read_text());raw=json.loads((run/'features.json').read_text())
rows={r['id']:r for r in data['samples']}; full={r['id']:r for r in raw['samples']}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'psycho-20261104-v1'})
dest=ROOT/'platform/site/demos/visuals'


def save(fig,name):
    fig.savefig(dest/(name+'.png'),dpi=170)
    svg=dest/(name+'.svg');fig.savefig(svg,metadata={'Date':None})
    svg.write_text('\n'.join(x.rstrip() for x in svg.read_text().splitlines())+'\n');plt.close(fig)


fig,axes=plt.subplots(2,2,figsize=(12,8)); colors=['#176a8a','#b85b26']
for name,label,color in zip(['muse_baseline','ref_vibeace'],['Muse','VibeAce'],colors):
    ids=['psy_'+name+'__'+s for s in ['anchor','am70_mild','am70_deep']]
    values=[rows[i]['psycho']['roughness70'] for i in ids]
    axes[0,0].plot([0,.25,.8],values,'o-',color=color,label=label)
    for x,y in zip([0,.25,.8],values):axes[0,0].annotate(f'{y:.3f}',(x,y),xytext=(0,7),textcoords='offset points',ha='center',color=color,fontsize=8)
    ids=['psy_'+name+'__'+s for s in ['anchor','tone4200_weak','tone4200_strong']]
    axes[0,1].plot([0,.08,.25],[rows[i]['psycho']['resonance_protected'] for i in ids],'o-',color=color,label=label)
axes[0,0].set(title='(a) 70 Hz AM: strength response',xlabel='Envelope modulation depth',ylabel='DW roughness at70 dB SPL (asper)',ylim=(0,3))
axes[0,1].set(title='(b) Added 4.2 kHz tone: strength response',xlabel='Added tone RMS / source RMS',ylabel='Harmonic-veto resonance proxy',ylim=(0,.65))
axes[0,0].legend(frameon=False);axes[0,1].legend(frameon=False)
names=['control_normal_synth','control_bright_synth','psy_control_fm_bell_style','psy_control_saw_bass_style']
labels=['Additive synth','Bright additive','FM bell','Saw bass']
val=[rows[n]['psycho']['roughness70'] for n in names]
axes[1,0].barh(labels,val,color='#65838b');axes[1,0].invert_yaxis()
for i,v in enumerate(val):axes[1,0].text(v+.035,i,f'{v:.3f}',va='center')
axes[1,0].axvline(rows['psy_ref_vibeace__am70_deep']['psycho']['roughness70'],color='#b85b26',ls='--',label='VibeAce deep AM')
axes[1,0].set(title='(c) Intended electronic timbres can be rough',xlabel='DW roughness at70 dB SPL (asper)',xlim=(0,4));axes[1,0].legend(frameon=False,fontsize=8)
y=np.arange(4)
for key,label,color,off in [('resonance_raw','Raw peak persistence','#9c6c52',-.13),('resonance_protected','With harmonic veto','#176a8a',.13)]:
    v=[rows[n]['psycho'][key] for n in names];axes[1,1].barh(y+off,v,height=.24,color=color,label=label)
    for yy,vv in zip(y+off,v):axes[1,1].text(vv+.007,yy,f'{vv:.4f}',va='center',fontsize=8)
axes[1,1].set_yticks(y,labels);axes[1,1].invert_yaxis();axes[1,1].set(title='(d) Harmonic veto still misses FM / high partials',xlabel='Relative resonance proxy (not defect probability)',xlim=(0,1.05));axes[1,1].legend(frameon=False,fontsize=8)
for ax in axes.flat:ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('Psychoacoustic probes: useful responses and timbre risks',fontsize=15,fontweight='bold')
fig.text(.5,.015,'Two matched source groups; controls are intended styles, not human naturalness gold. 70 dB = equal RMS pressure assumption.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.04,1,.95));save(fig,'psychoacoustic-response')

fig,axes=plt.subplots(2,2,figsize=(12,7),sharex='col')
ids=['psy_muse_baseline__anchor','psy_muse_baseline__am70_deep']; labels=['Matched anchor','Deep70 Hz AM']
def contribution(m):
    z=np.asarray(m['modulation_depth']);energy=np.asarray(m['band_energy_relative']);weight=energy**2*(energy>1e-3)
    return z*weight/np.maximum(weight.sum(axis=0,keepdims=True),1e-20)
maximum=max(np.quantile(contribution(full[i]['auditory']['maps']),.98) for i in ids)
for c,(rid,label) in enumerate(zip(ids,labels)):
    r=full[rid];m=r['auditory']['maps'];z=contribution(m);centers=m['center_hz']
    im=axes[0,c].imshow(z,aspect='auto',origin='lower',extent=[m['time_s'][0]-.125,m['time_s'][-1]+.125,0,32],vmin=0,vmax=maximum,cmap='viridis')
    ticks=[0,8,16,24,31];axes[0,c].set_yticks([k+.5 for k in ticks],[f'{centers[k]:.0f}' for k in ticks]);axes[0,c].set(ylabel='Gammatone band center (Hz)',title=label)
    q=r['roughness']['levels']['70'];axes[1,c].plot(q['time_s'],q['time_asper'],color='#176a8a');axes[1,c].set(xlabel='Time (s)',ylabel='DW roughness (asper)',ylim=(0,6),xlim=(0,10));axes[1,c].grid(alpha=.2)
fig.subplots_adjust(left=.07,right=.89,hspace=.25,top=.87,bottom=.12,wspace=.23)
cb=fig.colorbar(im,cax=fig.add_axes([.92,.55,.015,.31]));cb.set_label('Energy-weighted modulation contribution')
fig.suptitle('Auditory-band modulation and reference roughness over time',fontsize=14,fontweight='bold')
fig.text(.5,.02,'Same source; RMS matched; custom auditory map is not standardized roughness. DW shown at assumed70 dB SPL.',ha='center',fontsize=9)
save(fig,'psychoacoustic-time')
print('Two source-backed scientific figures written')
