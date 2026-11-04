"""Download only the fixed public inference artifacts used in this audit."""
import argparse
import json
from pathlib import Path
import urllib.request

from huggingface_hub import hf_hub_download
from signal_features import sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);args=p.parse_args()
    dest=Path(args.directory);dest.mkdir(parents=True,exist_ok=True)
    provenance=json.loads((Path(__file__).parent/'runs/2026-11-04/model-provenance.json').read_text())
    a=provenance['Audiobox'];path=dest/'audiobox-checkpoint.pt'
    if not path.exists(): urllib.request.urlretrieve(a['checkpoint_source'],path)
    if sha256(path)!=a['sha256']: raise ValueError('Audiobox weight hash mismatch')
    pam=provenance['PAM']
    path=hf_hub_download('microsoft/msclap','CLAP_weights_2023.pth',revision=pam['hf_revision'],local_dir=dest/'msclap')
    if sha256(path)!=pam['sha256']: raise ValueError('PAM weight hash mismatch')
    g=pam['gpt2_config_and_tokenizer']
    for name,record in g['files'].items():
        path=hf_hub_download(g['repo'],name,revision=g['revision'],local_dir=dest/'gpt2')
        if sha256(path)!=record['sha256']: raise ValueError('GPT2 config/tokenizer hash mismatch')
    print('Fixed weights/config/tokenizer hash-verified; upstream licenses still apply')


if __name__=='__main__':main()
