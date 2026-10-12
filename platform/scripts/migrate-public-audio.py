"""Convert legacy public media to MP3 with private backups and hash bindings."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / 'platform/site'
LEGACY = {'.wav', '.flac', '.ogg', '.m4a', '.aac', '.aiff', '.aif'}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backup', type=Path, required=True)
    args = parser.parse_args()
    if ROOT in args.backup.resolve().parents: raise ValueError('Backups must be outside repository')
    manifest = SITE / 'demos/audio-encoding.json'
    records = json.loads(manifest.read_text())['files'] if manifest.exists() else {}
    replacements = {}
    for source in sorted(SITE.rglob('*')):
        if source.suffix.lower() not in LEGACY: continue
        relative = source.relative_to(ROOT)
        backup = args.backup / relative
        backup.parent.mkdir(parents=True, exist_ok=True)
        source_hash = sha(source)
        if not backup.exists(): shutil.copy2(source, backup)
        if sha(backup) != source_hash: raise ValueError(f'Private backup mismatch: {source}')
        target = source.with_suffix('.mp3')
        if target.exists(): raise ValueError(f'Ambiguous preexisting MP3: {target}')
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(source), '-map', '0:a:0',
                        '-c:a', 'libmp3lame', '-b:a', '192k', '-ar', '48000',
                        '-map_metadata', '-1', str(target)], check=True)
        relative_mp3 = str(target.relative_to(ROOT))
        records[relative_mp3] = {'source_path': str(relative), 'source_audio_sha256': source_hash,
                                 'sha256': sha(target), 'bytes': target.stat().st_size,
                                 'encoding': 'MP3 192 kbps / 48 kHz; no loudness normalization'}
        replacements[source.name] = target.name
        source.unlink()
    # Mechanical migration of public path strings only; preserve scoring receipts in run folders.
    for path in SITE.rglob('*'):
        if path.suffix not in {'.html', '.js', '.json'} or path == manifest: continue
        original = path.read_text()
        text = original
        for before, after in replacements.items(): text = text.replace('/' + before, '/' + after)
        if text != original: path.write_text(text)
    manifest.write_text(json.dumps({'policy': 'public MP3 only; scored originals retained privately',
                                    'files': records}, indent=2) + '\n')
    print(json.dumps({'converted': len(replacements), 'source_bytes': sum((args.backup / r['source_path']).stat().st_size for r in records.values()),
                      'mp3_bytes': sum(r['bytes'] for r in records.values())}))


if __name__ == '__main__': main()
