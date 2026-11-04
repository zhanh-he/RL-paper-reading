"""Create an explicitly different scoring manifest from public MP3 copies.

This lets a fresh checkout exercise all scorers. It does NOT reproduce the
archived original-waveform scores exactly. No normalization, no synthetic labels.
"""
import argparse
import json
from pathlib import Path
import soundfile as sf
from signal_features import sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--audit');args=p.parse_args()
    here=Path(__file__).resolve().parent;root=here.parents[2]
    dest=Path(args.directory).resolve();dest.mkdir(parents=True,exist_ok=True)
    data=json.loads((Path(args.audit) if args.audit else here/'runs/2026-11-04/audit.json').read_text());rows=[]
    for r in data['samples']:
        source=root/'platform/site/demos'/r['audio']
        if sha256(source)!=r['mp3_sha256']:raise ValueError('Public MP3 changed')
        x,sr=sf.read(source,always_2d=True,dtype='float32')
        path=dest/(r['id']+'.wav');sf.write(path,x,sr,subtype='FLOAT')
        rows.append(dict(id=r['id'],path=str(path),excerpt_sha256=sha256(path),
                         public_mp3_sha256=sha256(source),kind=r['kind']))
    output=dest/'playback-manifest.json'
    output.write_text(json.dumps(dict(samples=rows,scope='MP3-decoded remeasurement; NOT original-waveform scores'),indent=2)+'\n')
    print(output)


if __name__=='__main__':main()
