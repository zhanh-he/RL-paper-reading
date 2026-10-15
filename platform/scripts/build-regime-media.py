"""Publish verified MP3 listening copies with the existing fixed-scale visuals."""
import argparse
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DEMO=ROOT/'platform/site/demos'


def load_module(name,filename):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(filename))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--staging',type=Path,required=True)
    p.add_argument('--additional-model',action='store_true');a=p.parse_args()
    report=json.loads((a.staging/'recovery.json').read_text())
    wave=load_module('regime_wave','render-wave-visuals.py');mel=load_module('regime_mel','render-mel-visuals.py')
    encpath=DEMO/'audio-encoding.json';encoding=json.loads(encpath.read_text())
    visuals=DEMO/'visuals/regimes';visuals.mkdir(parents=True,exist_ok=True)
    for name,binding in report['audio_files'].items():
        source=a.staging/'audio'/name
        assert name.endswith('.mp3') and hashlib.sha256(source.read_bytes()).hexdigest()==binding['sha256']
        assert source.stat().st_size==binding['bytes'] and len(binding['source_audio_sha256'])==64
        destination=DEMO/'audio'/name;shutil.copyfile(source,destination)
        encoding['files'][str(destination.relative_to(ROOT))]=binding
        if not (visuals/f'{source.stem}_wave.png').exists():wave.render(destination,visuals/f'{source.stem}_wave.png')
        if not (visuals/f'{source.stem}_spectrum.png').exists():mel.render(destination,visuals/f'{source.stem}_spectrum.png')
    for row in report['baseline']+report['long_replays']:
        stem=Path(row['asset']['audio']).stem
        row['asset'].update(wave=f'visuals/regimes/{stem}_wave.png',spectrum=f'visuals/regimes/{stem}_spectrum.png')
    for row in report['badcases']:
        for side in ['baseline','candidate']:
            stem=Path(row[side]['audio']).stem
            row[side].update(wave=f'visuals/regimes/{stem}_wave.png',spectrum=f'visuals/regimes/{stem}_spectrum.png')
    encpath.write_text(json.dumps(encoding,indent=2,ensure_ascii=False)+'\n')
    report_path=ROOT/'music-gen/reward-regime-study/recovery.json'
    existing=json.loads(report_path.read_text()) if report_path.exists() else {}
    if a.additional_model:
        assert existing['model']!=report['model']
        existing.setdefault('additional_models',{})[report['model']]=report
        existing['audio_files'].update(report['audio_files'])
        report=existing
    elif existing.get('additional_models'):
        report['additional_models']=existing['additional_models']
        for model in existing['additional_models'].values():report['audio_files'].update(model['audio_files'])
    report_path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(f'Published {len(report["audio_files"])} hash-verified MP3s; black-floor mel0-24kHz, fixed wave[-1,1].')


if __name__=='__main__':main()
