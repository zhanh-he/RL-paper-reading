"""Deterministic, unmodified excerpts from publisher-declared Suno V5 MP3s.

Sources JSON is produced by fetch_suno_real.py; raw/scoring audio stays private.
Selection is fixed before scoring: 10s at20/50/80% of the available start range.
No gain normalization, augmentation, feature-based mining or human labels.
"""
import argparse,hashlib,json,subprocess,re
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--sources',required=True);p.add_argument('--metadata-dir');p.add_argument('--output-dir',required=True);a=p.parse_args()
 sources=json.loads(Path(a.sources).read_text());metadata={}
 if a.metadata_dir:
  for lang in ['cn','en']:
   for item in json.loads((Path(a.metadata_dir)/(lang+'_metadata.json')).read_text())['rows']:
    metadata[Path(item['audio_path']).stem]=item
 out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);rows=[]
 for s in sources['sources']:
  source=Path(s['path']);assert sha(source)==s['sha256']
  probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_format','-show_streams','-of','json',str(source)]));duration=float(probe['format']['duration']);stream=probe['streams'][0]
  assert duration>=30
  prompt=s['id'].rsplit('_',1)[0]
  for idx,fraction in enumerate([.2,.5,.8]):
   start=round((duration-10)*fraction,3);rid=s['id']+f'__p{int(fraction*100)}';path=out/(rid+'.wav')
   subprocess.run(['ffmpeg','-v','error','-y','-i',str(source),'-ss',str(start),'-t','10','-ar','48000','-ac','2','-c:a','pcm_f32le',str(path)],check=True)
   rows.append(dict(id=rid,path=str(path.resolve()),excerpt_sha256=sha(path),source_original_sha256=s['sha256'],source_audio_filename=s['source_audio_filename'],source_revision=s['source_revision'],source_url=s['source_url'],source_shard=s['shard'],tar_member_offset=s['tar_member_offset'],tar_member_bytes=s['tar_member_bytes'],source_duration_s=duration,source_sample_rate=int(stream['sample_rate']),source_channels=stream['channels'],source_codec=stream['codec_name'],source_fraction=fraction,excerpt_start_s=start,duration_s=10,sample_rate=48000,channels=2,kind='real_suno_output',model='Suno V5 (dataset authors report)',checkpoint='original dataset audio; no RL update',input_id=s['id'],same_input_verified=False,source_family=prompt,pair_group=prompt+f'__p{int(fraction*100)}',song_id=s['id'],track_index=int(s['id'].rsplit('_',1)[1]),label=rid,selection_reason='Fixed20/50/80% positions of the available10s start range, chosen before scoring. No automatic defect label.',transform='none',rights='Muse dataset publisher declares MIT; SunoV5 provenance is author-reported. Source MP3 is already lossy.',license_url='https://huggingface.co/datasets/bolshyC/Muse/blob/'+s['source_revision']+'/README.md',human_rating=None))
 for row in rows:
  row['source_record']=dict(dataset='bolshyC/Muse',revision=row['source_revision'],shard=row['source_shard'],tar_member=row['source_audio_filename'],offset_bytes=row['tar_member_offset'],size_bytes=row['tar_member_bytes'])
  row['label']=row['song_id']+' · '+str(round(row['excerpt_start_s'],1))+'–'+str(round(row['excerpt_start_s']+10,1))+'s'
  if metadata:
   item=metadata[row['song_id']];text=''.join(re.sub(r'\W+','',section.get('text','').lower()) for section in item['sections'])
   row.update(publisher_style=item['style'],metadata_lyrics_sha256=hashlib.sha256(text.encode()).hexdigest(),metadata_style_sha256=hashlib.sha256(item['style'].encode()).hexdigest())
 checks=[]
 for family in sorted({r['source_family'] for r in rows}):
  members=[r for r in rows if r['source_family']==family and r['source_fraction']==.2]
  checks.append(dict(source_family=family,tracks=[r['song_id'] for r in members],same_published_style=len({r.get('metadata_style_sha256',r['song_id']) for r in members})==1,same_normalized_lyrics=len({r.get('metadata_lyrics_sha256',r['song_id']) for r in members})==1,same_input_verified=False))
 d=dict(schema_version=1,metadata_pair_checks=checks,selection=sources['selection'],source_dataset_revision=sources['dataset_revision'],samples=rows)
 (out.parent/'suno_manifest.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(songs=len(sources['sources']),clips=len(rows))))
if __name__=='__main__':main()
