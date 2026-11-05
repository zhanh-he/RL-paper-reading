"""Fixed real-output windows, original levels; MP3 listening/full-song copies."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import numpy as np,soundfile as sf

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--manifest',required=True);a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);clips=[];tracks=[]
 for runfile in a.runs:
  run=json.loads(Path(runfile).read_text());assert run['status']=='complete'
  for row in run['samples']:
   path=Path(row['path']);assert sha(path)==row['audio_sha256'];wave,sr=sf.read(path,dtype='float32',always_2d=True);duration=len(wave)/sr;assert sr==48000 and duration>=10 and np.isfinite(wave).all()
   full=out/(row['id']+'__full.mp3')
   if not full.exists():subprocess.run(['ffmpeg','-v','error','-y','-i',str(path),'-ar','48000','-c:a','libmp3lame','-b:a','192k',str(full)],check=True)
   track=row|dict(full_mp3_path=str(full),full_mp3_sha256=sha(full),actual_duration_s=duration)
   tracks.append(track)
   starts=[10.,40.] if duration>=50 else [round((duration-10)/2,3)]
   for wi,start in enumerate(starts):
    rid=row['id']+f'__w{wi+1}';wav=out/(rid+'.wav');audio=wave[round(start*sr):round((start+10)*sr)];assert len(audio)==480000
    sf.write(wav,audio,sr,subtype='FLOAT');mp3=out/(rid+'.mp3')
    if not mp3.exists():subprocess.run(['ffmpeg','-v','error','-y','-i',str(wav),'-ar','48000','-c:a','libmp3lame','-b:a','192k',str(mp3)],check=True)
    clips.append(dict(id=rid,path=str(wav),excerpt_sha256=sha(wav),source_original_sha256=row['audio_sha256'],source_case_id=row['id'],source_model=row.get('model',run['profile']['model']),model=row.get('model',run['profile']['model']),family=row.get('family'),arm=row.get('arm'),seed=row.get('seed'),duration_s=10.,sample_rate=48000,channels=audio.shape[1],excerpt_start_s=start,source_duration_s=duration,window_index=wi+1,pair_group=str(row.get('family'))+f'_s{row.get("seed")}_w{wi+1}',source_run_sha256=sha(runfile),mp3_path=str(mp3),mp3_sha256=sha(mp3),full_mp3_path=str(full),full_mp3_sha256=sha(full),human_rating=None,kind='actual_generated_control',selection_reason='Fixed10–20s and40–50s for outputs>=50s; otherwise centered10s. Rule fixed before feature scoring; no gain normalization.'))
 d=dict(schema_version=1,selection='Fixed real output windows; no added artifacts or gain normalization',tracks=tracks,samples=clips);Path(a.manifest).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(tracks=len(tracks),clips=len(clips))))
if __name__=='__main__':main()
