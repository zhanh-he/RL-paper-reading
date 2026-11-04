"""Train a small Bradley–Terry head on frozen audio embeddings and REAL labels.

No head is shipped: the pilot has no human labels. npz must contain ids and
embeddings. pairs JSON uses {pairs:[{a,b,choice:'a'|'b'|'tie',split:'train'|'test'}]}.
Split by source song and prompt BEFORE annotations/augmentation, not random clips.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def predict(embedding, model, *, encoder_id, encoder_revision):
    if model['encoder_id']!=encoder_id or model['encoder_revision']!=encoder_revision:
        raise ValueError('Head and encoder identity mismatch')
    z=np.asarray(embedding,dtype=float); w=np.asarray(model['weights'],dtype=float)
    if z.ndim!=1 or z.shape!=w.shape or not np.isfinite(z).all() or not np.isfinite(w).all() or np.linalg.norm(z)<1e-8:
        raise ValueError('Invalid head or embedding')
    return float((z/np.linalg.norm(z))@w)


def train(embeddings, pairs, l2=1.):
    train_pairs=[r for r in pairs if r['split']=='train' and r['choice'] in ('a','b','tie')]
    if len(train_pairs)<10:
        raise ValueError('Need at least 10 genuine training comparisons; more independent songs recommended')
    z=np.array([embeddings[r['a']]-embeddings[r['b']] for r in train_pairs])
    y=np.array([{'a':1.,'b':0.,'tie':.5}[r['choice']] for r in train_pairs])
    def loss(w):
        v=z@w
        return (np.logaddexp(0,v)-y*v).mean()+l2*np.dot(w,w)/2
    def jac(w): return z.T@(expit(z@w)-y)/len(y)+l2*w
    result=minimize(loss,np.zeros(z.shape[1]),jac=jac,method='L-BFGS-B')
    if not result.success: raise RuntimeError(result.message)
    return result.x


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--embeddings',required=True)
    p.add_argument('--pairs',required=True)
    p.add_argument('--split-registry',required=True,help='JSON samples: id -> {song_group,prompt_group,split}')
    p.add_argument('--output',required=True)
    p.add_argument('--encoder-id',required=True)
    p.add_argument('--encoder-revision',required=True)
    p.add_argument('--l2',type=float,default=1.)
    args=p.parse_args()
    if args.l2<=0: raise ValueError('l2 must be positive')
    data=np.load(args.embeddings,allow_pickle=False)
    zs=data['embeddings']; ids=data['ids'].tolist()
    if zs.ndim!=2 or len(ids)!=len(zs) or len(ids)!=len(set(ids)):
        raise ValueError('Need unique IDs aligned with a 2D embedding matrix')
    norms=np.linalg.norm(zs,axis=1,keepdims=True)
    if not np.isfinite(zs).all() or np.any(norms<1e-8): raise ValueError('Invalid embedding')
    embeddings=dict(zip(ids,zs/norms))
    pairs=json.loads(Path(args.pairs).read_text())['pairs']
    registry=json.loads(Path(args.split_registry).read_text())['samples']
    endpoints={split:set() for split in ('train','test')}
    for row in pairs:
        if row['split'] not in ('train','test'): raise ValueError('Unknown split')
        if row['choice'] not in ('a','b','tie'): raise ValueError('Unknown preference choice')
        for rid in (row['a'],row['b']):
            if rid not in embeddings or rid not in registry: raise ValueError('Missing embedding or source registry')
            item=registry[rid]
            if item['split']!=row['split'] or not item['song_group'] or not item['prompt_group']:
                raise ValueError('Pair disagrees with independently prepared source split registry')
            endpoints[row['split']].add(rid)
    if endpoints['train']&endpoints['test']: raise ValueError('Audio ID leakage')
    for key in ('song_group','prompt_group'):
        groups={split:{registry[rid][key] for rid in endpoints[split]} for split in ('train','test')}
        if groups['train']&groups['test']: raise ValueError(key+' leaks across train and test')
    w=train(embeddings,pairs,args.l2)
    test=[r for r in pairs if r['split']=='test' and r['choice'] in ('a','b')]
    margins=[float((embeddings[r['a']]-embeddings[r['b']])@w) for r in test]
    correct=sum(.5 if abs(margin)<1e-12 else float((margin>0)==(r['choice']=='a')) for r,margin in zip(test,margins))
    result=dict(version='bt-linear-v1',encoder_id=args.encoder_id,encoder_revision=args.encoder_revision,
                l2=args.l2,weights=w.tolist(),embedding_normalization='L2',
                train_pairs=sum(r['split']=='train' for r in pairs),test_pairs=len(test),
                test_accuracy=correct/len(test) if test else None,
                calibration=None,status='trained; calibration and held-out acceptance still required')
    Path(args.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__': main()
