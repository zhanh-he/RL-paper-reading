"""Prepare native G2P dictionaries on an internet-enabled preparation node."""
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile


def nltk_assets(root):
    root = Path(root)/'nltk_data'; records=[]
    for group, name in [('corpora','cmudict'), ('taggers','averaged_perceptron_tagger'),
                        ('taggers','averaged_perceptron_tagger_eng')]:
        folder=root/group; folder.mkdir(parents=True,exist_ok=True)
        archive=folder/f'{name}.zip'
        url=f'https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/{group}/{name}.zip'
        if not archive.exists():
            temporary=archive.with_suffix('.tmp')
            with urllib.request.urlopen(url,timeout=120) as source, temporary.open('wb') as target:
                while block:=source.read(1<<20): target.write(block)
            temporary.replace(archive)
        with zipfile.ZipFile(archive) as data:
            if any('..' in Path(n).parts or Path(n).is_absolute() for n in data.namelist()):
                raise ValueError('Unsafe archive entry')
            data.extractall(folder)
        records.append({'name':name,'url':url,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()})
    (root/'receipt.json').write_text(json.dumps(records,indent=2)+'\n')


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    nltk_assets(parser.parse_args().root)
