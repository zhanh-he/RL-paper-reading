"""Export an MP3-only public blind deck and keep its unblinding key separately."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
from itertools import combinations
import json
from pathlib import Path
import random
import re
import secrets
import shutil
import subprocess


BATCH = 'singer-blind-dev-20261117-v1'
TARGET_PAIRS = 64


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for data in iter(lambda: source.read(1 << 20), b''):
            digest.update(data)
    return digest.hexdigest()


def choose_extra(groups, old_keys, used, rng):
    pool = [ab for g in groups.values() for ab in combinations(sorted(g), 2) if frozenset(ab) not in old_keys]
    rng.shuffle(pool)
    unseen = {x for g in groups.values() for x in g} - used
    added = []
    while unseen:
        best = max(pool, key=lambda ab: sum(x in unseen for x in ab))
        if not unseen.intersection(best):
            raise ValueError('Cannot cover all predeclared candidates')
        pool.remove(best)
        unseen.difference_update(best)
        added.append((*best, 'coverage_expansion'))
    return added, pool


def prepare(study, output):
    protocol = study/'protocol'
    families = {r['family_id']: r for r in map(json.loads, (protocol/'prompt_seeds.jsonl').read_text().splitlines())
                if r['split']=='dev'}
    all_rows = read(study/'solo-manifest.json')
    rows = {r['rollout_id']:r for r in all_rows if r['split']=='dev'}
    scores = {r['rollout_id']:r for r in read(study/'solo-scores.json') if r['rollout_id'] in rows}
    if len(rows)!=96 or len(families)!=12 or len(scores)!=96 or len(all_rows)!=160:
        raise ValueError('Unverified development cohort')

    groups = defaultdict(list)
    published_condition = {}
    for id,row in rows.items():
        family = families[row['family_id']]
        task = row['task_id']
        if task not in ('solo_male','solo_female') or task not in family['tasks']:
            raise ValueError('Unexpected task')
        lyrics = '\n\n'.join(f'[{tag}]\n'+'\n'.join(s['lines']) for tag,s in zip(('Verse','Chorus','Bridge'),family['sections']))
        sc = scores[id]
        voice = task.removeprefix('solo_')
        style = f"{family['style']}; one {voice}-sounding lead singer, same singer throughout, one lead vocal layer, no backing vocals or doubling; allow natural register and dynamics"
        request = row['actual_request']
        if row['lyrics']!=lyrics or request['lyrics']!=lyrics or row['style']!=style or request['style']!=style or request['abc'] is not None or request['cot']!='off':
            raise ValueError('Only the published original prompts may appear in the blind deck')
        if row['generation_status']!='complete' or sc['scoring_status']!='complete' or row['mix_sha256']!=sc['mix_sha256']:
            raise ValueError('Unverified mix binding')
        groups[row['group_id']].append(id)
        published_condition[id] = dict(style=style, lyrics=lyrics)
    if len(groups)!=24 or any(sorted(rows[id]['generation_seed'] for id in g)!=[101,307,587,911] for g in groups.values()):
        raise ValueError('Expected 24 complete groups')
    prior = read(study/'blind-dev-private-mapping.json')
    listening = read(study/'blind-dev/listening-manifest.json')['clips']
    key = dict(batch_id=BATCH, clips=prior['clips'], pairs=prior['pairs'])
    if len(key['pairs'])!=32 or set(key['clips'])!=set(listening):
        raise ValueError('Prior blind deck is missing')
    ids = {}
    for b,entry in key['clips'].items():
        if not re.fullmatch(r'L[A-F0-9]{8}', b) or entry['rollout_id'] not in rows:
            raise ValueError('Invalid prior blind ID')
        original = rows[entry['rollout_id']]
        if original['mix_sha256']!=entry['mix_sha256']:
            raise ValueError('Prior blind ID maps to different bytes')
        ids.setdefault(entry['rollout_id'], b)
    old_keys = {frozenset(key['clips'][p[k]]['rollout_id'] for k in ('first','second')) for p in key['pairs']}
    rng = random.Random(20261118)
    added,pool = choose_extra(groups,old_keys,set(ids),rng)
    needed = TARGET_PAIRS-len(key['pairs'])-len(added)
    if needed<0 or needed>len(pool):
        raise ValueError('Deck size cannot cover the frozen cohort')
    added.extend((*ab,'random_expansion') for ab in rng.sample(pool,needed))
    for a,b,selection in added:
        shown=[]
        for id in (a,b):
            if id not in ids:
                alias='L'+secrets.token_hex(4).upper()
                while alias in key['clips']:
                    alias='L'+secrets.token_hex(4).upper()
                ids[id]=alias
                row=rows[id]
                key['clips'][alias]={k:row[k] for k in ('rollout_id','family_id','group_id','conditioning_id','split','task_id','mix_sha256')}
            shown.append(ids[id])
        if rng.randrange(2):
            shown.reverse()
        key['pairs'].append(dict(id=f"P{len(key['pairs'])+1:02d}",first=shown[0],second=shown[1],selection=selection))
    if set(ids)!=set(rows) or len(key['pairs'])!=TARGET_PAIRS:
        raise ValueError('Incomplete expanded deck')

    output.mkdir(mode=0o700)
    assets=output/'audio'
    assets.mkdir()

    def encode(item):
        alias,entry=item
        id=entry['rollout_id']; row=rows[id]; score=scores[id]
        files={}
        for kind,suffix,src,expected in [('mix','',row['mix_path'],row['mix_sha256']),
                                         ('vocal','-v',score['vocal_path'],score['vocal_sha256'])]:
            path=Path(src)
            if sha(path)!=expected:
                raise ValueError('Archived audio changed')
            dest=assets/f'{alias}{suffix}.mp3'
            if alias in listening:
                prior=study/'blind-dev/audio'/dest.name
                if sha(prior)!=listening[alias][kind+'_sha256']:
                    raise ValueError('Prior listening copy changed')
                shutil.copyfile(prior,dest)
            else:
                subprocess.run(['ffmpeg','-nostdin','-v','error','-threads','1','-i',str(path),'-map_metadata','-1',
                                '-map_chapters','-1','-vn','-codec:a','libmp3lame','-b:a','192k','-ar','48000',str(dest)],check=True)
            files[kind]=dict(file=dest.name,sha256=sha(dest),bytes=dest.stat().st_size)
        return alias,dict(duration_s=row['duration_s'],**files,**published_condition[id])

    with ThreadPoolExecutor(max_workers=6) as pool:
        clips=dict(pool.map(encode,sorted(key['clips'].items())))
    pairs=[]
    for p in key['pairs']:
        a,b=(key['clips'][p[k]] for k in ('first','second'))
        if a['group_id']!=b['group_id'] or a['rollout_id']==b['rollout_id']:
            raise ValueError('Pair is not independently sampled from the same condition')
        pairs.append({k:p[k] for k in ('id','first','second')})
    pub=dict(schema=1,batch_id=BATCH,unique_recordings=len(rows),condition_count=len(groups),
             family_count=len(families),clips=clips,pairs=pairs)
    write(output/'deck.json',pub)
    fields=(protocol/'annotations.csv').read_text().splitlines()[0].split(',')
    with (output/'annotations.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,lineterminator='\n')
        writer.writeheader()
        writer.writerows(dict(batch_id=BATCH,label_type='clip',blind_id=x) for x in clips)
        writer.writerows(dict(batch_id=BATCH,label_type='pair',blind_id=p['first'],other_blind_id=p['second']) for p in pairs)
    write(study/f'{BATCH}-private-mapping.json',key)
    print(BATCH,len(clips),len(rows),len(pairs),sum(f.stat().st_size for f in assets.iterdir()),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    prepare(a.study,a.out)
