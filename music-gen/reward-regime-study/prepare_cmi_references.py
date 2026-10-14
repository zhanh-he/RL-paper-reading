"""Download the frozen300 native reference inputs privately, with byte receipts."""
import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import hf_hub_download
import soundfile as sf

REVISION = '5282fbe784e326014894b299cb22645cd7d56057'
SOURCE_SHA = 'fd157070f09ad51d807f4b447bf5a9668fd9b00890c7d646ffd1331850e7d8d0'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(); args.root.mkdir(exist_ok=True, parents=True)
    from prepare_frontend_resources import nltk_assets
    nltk_assets(args.root)
    data = json.loads((args.root/'data.json').read_text())
    ids = {r['condition_id'] for part in ['train', 'valid'] for r in data[part]}
    source = hf_hub_download('HaiwenXia/cmi-pref', 'cmi_train.jsonl', repo_type='dataset', revision=REVISION,
                             local_dir=args.root/'reference-source')
    if sha(source) != SOURCE_SHA: raise ValueError('CMI metadata hash mismatch')
    references = {}; norm = lambda text: ' '.join(text.split()).casefold()
    for line in Path(source).open(encoding='utf-8'):
        row = json.loads(line)
        key = (norm(row['prompt']), norm(row['lyrics']), row['ref-audio-path'].strip())
        cid = hashlib.sha256(json.dumps(key, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
        if cid not in ids or cid in references: continue
        filename = key[2].removeprefix('cmi-pref/')
        path = hf_hub_download('HaiwenXia/cmi-pref', filename, repo_type='dataset', revision=REVISION,
                               local_dir=args.root/'references')
        info = sf.info(path)
        if info.duration <= 0: raise ValueError('Empty reference')
        references[cid] = {'path': path, 'sha256': sha(path), 'seconds': info.duration,
                           'source': filename, 'revision': REVISION}
        print(f'Native references: {len(references)}/300', flush=True)
    if references.keys() != ids: raise ValueError('Incomplete reference coverage')
    (args.root/'references.json').write_text(json.dumps(references, indent=2)+'\n')


if __name__ == '__main__': main()
