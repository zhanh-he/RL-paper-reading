"""Export hash-bound MP3 listening copies, without normalization.

Never place the manifest's private source paths in public output. Only the
explicit allowlist is copied. Run this from the repository after scoring.
"""
import argparse
import json
from pathlib import Path
import subprocess

from signal_features import sha256


PUBLIC_FIELDS = ['id','label','model','checkpoint','input_id','pair_group','kind','source_asset',
                 'source_original_sha256','source_sample_rate','source_duration','source_channels',
                 'source_encoding','source_url','credit','license_url','license_metadata_url',
                 'rights','selection_reason','source_record','public_source_sha256','excerpt_sha256',
                 'duration_s','sample_rate','offset_s','transform','transform_description','base_id','matched_comparator_id']


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True)
    p.add_argument('--repo',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    root=Path(args.repo).resolve()
    dest=root/'platform/site/demos/audio/naturalness-20261104';dest.mkdir(parents=True,exist_ok=True)
    bindings_path=root/'platform/site/demos/audio-encoding.json'
    bindings=json.loads(bindings_path.read_text());rows=[]
    for row in json.loads(Path(args.manifest).read_text())['samples']:
        expected=row['excerpt_sha256']
        if sha256(row['path'])!=expected: raise ValueError('Scoring audio changed: '+row['id'])
        path=dest/(row['id']+'.mp3')
        subprocess.run(['ffmpeg','-v','error','-y','-i',row['path'],'-ar','48000',
                        '-c:a','libmp3lame','-b:a','192k',str(path)],check=True)
        hash=sha256(path)
        binding=dict(source_audio_sha256=expected,sha256=hash,bytes=path.stat().st_size,
                     encoding='MP3 192 kbps / 48 kHz; no loudness normalization',
                     source_path='Private archived scoring excerpt; identity is excerpt_sha256')
        bindings['files'][str(path.relative_to(root))]=binding
        public={k:row[k] for k in PUBLIC_FIELDS if k in row}
        public.update(audio=str(path.relative_to(root/'platform/site/demos')),mp3_sha256=hash,
                      listening_copy='MP3 transcode; scores refer to the archived float WAV excerpt, not this lossy file',
                      human_rating=None)
        rows.append(public)
    bindings_path.write_text(json.dumps(bindings,ensure_ascii=False,indent=2)+'\n')
    Path(args.output).write_text(json.dumps({'schema_version':1,'samples':rows},ensure_ascii=False,indent=2)+'\n')
    print(f'{len(rows)} MP3 copies and separate source/excerpt/listening hashes')


if __name__=='__main__': main()
