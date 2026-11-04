"""Bounded HTTP ranges: first two actual MP3 members of eight fixed public shards."""
import urllib.request,json,hashlib,tarfile,io,concurrent.futures,time,argparse
from pathlib import Path
REV='b1bf3bf906daab3a896e14f6dea58cc295848452'
BASE=f'https://huggingface.co/datasets/bolshyC/Muse/resolve/{REV}/'
OUT=None # supplied by --output-dir; keep raw audio outside the repository
SHARDS=[f'cn_part{i:02}_of_25.tar' for i in [1,9,17,25]]+[f'en_part{i:02}_of_35.tar' for i in [1,12,23,35]]
def read_range(shard,start,end):
 for attempt in range(3):
  try:
   u=BASE+shard+f'?range={start}-{end}'
   req=urllib.request.Request(u,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'})
   with urllib.request.urlopen(req,timeout=90) as r:
    data=r.read(end-start+2);cr=r.headers.get('Content-Range','')
    if r.status!=206 or not cr.startswith(f'bytes {start}-{end}/') or len(data)!=end-start+1:raise ValueError((r.status,cr,len(data),start,end))
    return data
  except Exception:
   if attempt==2:raise
   time.sleep(2)
def one(shard):
 pos=0;rows=[];limit=0
 while len(rows)<2:
  header=read_range(shard,pos,pos+511);info=tarfile.TarInfo.frombuf(header,'utf-8','surrogateescape');data_pos=pos+512
  if info.isfile() and info.name.endswith('.mp3'):
   if not 0<info.size<=30_000_000:raise ValueError('unexpected size')
   if Path(info.name).name!=info.name or info.name.startswith('.'):raise ValueError('expected flat safe MP3 member name')
   path=OUT/info.name
   audio=read_range(shard,data_pos,data_pos+info.size-1);path.write_bytes(audio)
   row=dict(id=path.stem,source_audio_filename=info.name,shard=shard,source_url=BASE+shard,source_revision=REV,tar_member_offset=data_pos,tar_member_bytes=info.size,sha256=hashlib.sha256(audio).hexdigest(),path=str(path),provenance='Muse published SunoV5 training audio; model version is author-reported',license='MIT (dataset publisher declaration)',human_label=None)
   rows.append(row);print(json.dumps(dict(id=row['id'],bytes=info.size,sha256=row['sha256'])),flush=True)
  pos=data_pos+((info.size+511)//512)*512;limit+=1
  if limit>10:raise ValueError('too many non-audio tar members')
 return rows
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--output-dir',required=True);parser.add_argument('--with-metadata',action='store_true');args=parser.parse_args();OUT=Path(args.output_dir);OUT.mkdir(parents=True,exist_ok=True)
 rows=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
  for v in ex.map(one,SHARDS):rows.extend(v)
 (OUT/'sources.json').write_text(json.dumps(dict(selection='First two MP3 members of fixed CN shards1/9/17/25 and EN1/12/23/35, selected before scoring; not representative random population sample',dataset_revision=REV,sources=rows),indent=2))

 if args.with_metadata:
  def metadata(lang):
   wanted={r['id'].rsplit('_',1)[0] for r in rows if r['id'].startswith('suno_'+lang+'_')};found=[];n=0;size=0
   for split in ['train','val']:
    with urllib.request.urlopen(BASE+split+'_'+lang+'.jsonl',timeout=90) as stream:
     for line in stream:
      n+=1;size+=len(line);item=json.loads(line)
      if item.get('song_id') in wanted:found.append(item)
   (OUT/(lang+'_metadata.json')).write_text(json.dumps(dict(rows=found,lines_scanned=n,bytes_streamed=size)))
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:list(ex.map(metadata,['cn','en']))
