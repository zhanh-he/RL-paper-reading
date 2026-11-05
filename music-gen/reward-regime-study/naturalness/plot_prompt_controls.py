"""Measured fixed-window controls; no perceptual score or detection accuracy."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 p=argparse.ArgumentParser();p.add_argument('--audit',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args();d=json.loads(Path(a.audit).read_text());out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);tracks={r['id']:r for r in d['tracks']}
 plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
 metrics=[('modulation_depth','Auditory modulation depth'),('roughness70','DW70 mean [asper]'),('resonance_protected','Harmonic-veto peak proxy'),('PQ','Audiobox PQ (quality proxy)')]
 for mode in ['diff-cfg','muse-renderer']:
  fig,axes=plt.subplots(1,4,figsize=(13,3.7),layout='constrained')
  for ax,(key,label) in zip(axes,metrics):
   if mode=='diff-cfg':
    for family,color in [('acoustic','#117f76'),('synth','#ba6934')]:
     rr=[tracks[f'diff_{family}_s37_'+('base' if v==2 else 'cfg'+str(v))] for v in [0,1,2]];values=[r.get('audiobox_window_mean_PQ') if key=='PQ' else r['window_mean'][key] for r in rr];ax.plot([0,1,2],values,'o-',label=family,color=color)
    ax.set_xlabel('Native CFG strength');ax.set_xticks([0,1,2]);ax.legend(frameon=False)
   else:
    rr=[tracks['muse_synth_s37_base_used'+str(v).replace('.','p')] for v in [1.,1.5,2.]];values=[r.get('audiobox_window_mean_PQ') if key=='PQ' else r['window_mean'][key] for r in rr];ax.plot([1,1.5,2],values,'o-',color='#117f76');ax.set_xlabel('Actual decoder guidance');ax.set_xticks([1,1.5,2])
   ax.set_ylabel(label);ax.grid(alpha=.15)
  title='DiffRhythm2: same lyrics / style / initial noise, seed37; fixed 10 s windows; standard CFG scale =1+native strength' if mode=='diff-cfg' else 'Muse renderer: exact same 750 audio codes and decoder noise; centered 10 s of real 30 s output'
  fig.suptitle(title+'\nIndependent axes; descriptor movement is not a human naturalness verdict',fontsize=10.5)
  for ext in ['png','svg']:fig.savefig(out/('prompt-control-'+mode+'.'+ext),dpi=180)
  plt.close(fig)
 for path in out.glob('prompt-control-*.svg'):path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
if __name__=='__main__':main()
