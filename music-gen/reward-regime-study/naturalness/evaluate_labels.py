"""Evaluate exported REAL listener labels; absent labels produce null results.

Example: python evaluate_labels.py --audit runs/2026-11-04/audit.json
 --labels listener1.json listener2.json --output agreement.json
Scores are not copied into human columns. Perturbations are not gold labels.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

METRICS={'Audiobox_PQ':('audiobox','PQ'),'PAM':('pam','score'),
         'SongEval_Naturalness':('songeval','Naturalness'),
         'SongEval_Musicality':('songeval','Musicality'),
         'Negative_Auditory_Modulation':('psycho','modulation_depth'),
         'Negative_DW_Roughness70':('psycho','roughness70'),
         'Negative_Resonance_Protected':('psycho','resonance_protected')}


def metric(row,path):
    try: value=row[path[0]][path[1]]
    except (KeyError,TypeError): return None
    return float(value) if value is not None and np.isfinite(value) else None


def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',required=True)
    p.add_argument('--labels',nargs='*',default=[]);p.add_argument('--output',required=True)
    args=p.parse_args();audit=json.loads(Path(args.audit).read_text())
    samples={r['id']:r for r in audit['samples']}; ratings=[];pairs=[]
    for file in args.labels:
        labels=json.loads(Path(file).read_text())
        if labels['schema']!='rl-lab-naturalness-labels-v1' or labels['dataset_version']!=audit['metadata']['version']:
            raise ValueError('Wrong label schema or dataset version')
        for r in labels['ratings']:
            if r['sample_id'] not in samples: raise ValueError('Unknown sample')
            for key in ['artificialness','quality','musicality']:
                if r.get(key) is not None and (not isinstance(r[key],int) or not 1<=r[key]<=5): raise ValueError('Invalid real rating')
            if r.get('artificialness') is not None: ratings.append(r)
        for pair in labels['pairs']:
            if pair['a'] not in samples or pair['b'] not in samples: raise ValueError('Unknown pair endpoint')
            if pair['a']!=pair['b'] and pair.get('naturalness') in ['A','B','tie','uncertain']: pairs.append(pair)
    result=dict(human_ratings_count=len(ratings),human_pairs_count=len(pairs),
                status='exploratory_real_labels' if ratings or pairs else 'no_human_labels',
                scores={},warning='Same-song/transform observations are dependent. Few groups do not establish generalization.')
    for name,path in METRICS.items():
        direction=-1 if name.startswith('Negative_') else 1
        ids={r['sample_id'] for r in ratings}; x=[];y=[]
        for rid in ids:
            value=metric(samples[rid],path)
            if value is None: continue
            x.append(direction*value);y.append(float(np.median([6-r['artificialness'] for r in ratings if r['sample_id']==rid])))
        rho=float(spearmanr(x,y).statistic) if len(x)>=3 and np.ptp(x)>0 and np.ptp(y)>0 else None
        judged=[];family_judged=[]
        for r in pairs:
            if r['naturalness'] not in ['A','B']: continue
            a,b=samples[r['a']],samples[r['b']]
            # Keep mismatched content judgments available separately in the exported labels.
            same=(a.get('same_input_verified') is not False and b.get('same_input_verified') is not False) and ((a.get('input_id') and a.get('input_id')==b.get('input_id')) or a.get('base_id')==b['id'] or b.get('base_id')==a['id'])
            av,bv=metric(a,path),metric(b,path)
            if av is not None and bv is not None and a.get('kind')=='real_suno_output' and b.get('kind')=='real_suno_output' and a.get('source_family')==b.get('source_family'):
                da,db=direction*av,direction*bv
                family_judged.append((a['source_family'],.5 if abs(da-db)<1e-12 else float((da>db)==(r['naturalness']=='A'))))
            if not same or av is None or bv is None: continue
            av*=direction;bv*=direction
            correct=.5 if abs(av-bv)<1e-12 else float((av>bv)==(r['naturalness']=='A'))
            group=a.get('source_family') or a.get('input_id') or a.get('source_original_sha256') or a.get('pair_group') or a['id']
            judged.append((group,correct))
        groups=sorted({g for g,_ in judged}); ci=None
        if len(groups)>=3:
            rng=np.random.default_rng(20261104)
            grouped={g:[v for k,v in judged if k==g] for g in groups}
            draws=[np.mean([v for g in rng.choice(groups,len(groups)) for v in grouped[g]]) for _ in range(2000)]
            ci=np.quantile(draws,[.025,.975]).tolist()
        family_groups=sorted({g for g,_ in family_judged});family_ci=None
        if len(family_groups)>=3:
            rng=np.random.default_rng(20261104);fg={g:[v for k,v in family_judged if k==g] for g in family_groups}
            family_ci=np.quantile([np.mean([v for g in rng.choice(family_groups,len(family_groups)) for v in fg[g]]) for _ in range(2000)],[.025,.975]).tolist()
        result['scores'][name]=dict(real_suno_family_decisive_pairs=len(family_judged),real_suno_family_groups=len(family_groups),real_suno_family_pair_accuracy=float(np.mean([v for _,v in family_judged])) if family_judged else None,real_suno_family_ci95=family_ci,real_suno_family_caveat='Exploratory human preferences within source families; generation inputs and musical content are not proven matched.',direction=direction,direction_note='Hypothesis for evaluation only; not validated reward',rated_unique_samples=len(x),spearman_naturalness=rho,
                                  comparable_decisive_pairs=len(judged),independent_source_groups=len(groups),
                                  pair_accuracy=float(np.mean([v for _,v in judged])) if judged else None,
                                  group_bootstrap_ci95=ci,
                                  human_ties=sum(r.get('naturalness')=='tie' for r in pairs),
                                  human_uncertain=sum(r.get('naturalness')=='uncertain' for r in pairs))
    Path(args.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__': main()
