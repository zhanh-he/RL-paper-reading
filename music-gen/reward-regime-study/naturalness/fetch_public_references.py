"""Fetch two attributed librosa reference recordings (not naturalness gold)."""
import argparse
import json
from pathlib import Path

import librosa
import soundfile as sf

from signal_features import sha256


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--directory', required=True)
    p.add_argument('--output', required=True)
    args=p.parse_args()
    dest=Path(args.directory).resolve(); dest.mkdir(parents=True,exist_ok=True)
    rows=[]
    for key, offset, label, author, origin in [
        ('vibeace',10,'录制参考 · Vibe Ace 器乐组合','Kevin MacLeod','https://freemusicarchive.org/music/Kevin_MacLeod/Jazz_Sampler/Vibe_Ace'),
        ('fishin',30,'录制参考 · Let’s Go Fishin’ 民谣歌曲','Karissa Hobbs','https://freemusicarchive.org/music/Karissa_Hobbs/Age_of_Flowers/09_Lets_Go_Fishin'),
    ]:
        source=Path(librosa.example(key,hq=True))
        # Cache exact downloaded .ogg privately; decoded excerpt is not a lossless master.
        info=sf.info(source)
        license_text=source.with_suffix('.txt')
        librosa.util.example_info(key)
        from shutil import copyfile
        copied=dest/source.name; copyfile(source,copied)
        rows.append(dict(id='ref_'+key,label=label,path=str(copied),model='recorded music reference',checkpoint='not applicable',
                         input_id='librosa example '+key, pair_group=None,source_asset='https://librosa.org/data/audio/'+source.name,
                         source_url=origin,credit=author,source_sample_rate=info.samplerate,source_duration=info.duration,
                         source_original_sha256=sha256(copied),source_encoding='OGG Vorbis; lossy distribution source',
                         offset_s=offset,kind='recorded_reference',rights='CC-BY-3.0',
                         license_url='https://creativecommons.org/licenses/by/3.0/',
                         license_metadata_url='https://librosa.org/data/audio/'+source.name.replace('.hq.ogg','.txt'),
                         selection_reason='Recorded music / acoustic-instrument reference; recording is not a human naturalness label'))
    Path(args.output).write_text(json.dumps({'samples':rows},ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__': main()
