"""Build isolated native-model environments without modifying shared scorers."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import venv

PACKAGES = {
    'diffrhythm2': ['transformers==4.47.1', 'tokenizers==0.21.4', 'torchdiffeq==0.2.5', 'diffusers==0.35.1',
                   'phonemizer', 'espeakng-loader', 'pyopenjtalk', 'pykakasi', 'py3langid==0.2.2', 'pypinyin',
                   'cn2an', 'onnxruntime', 'jieba', 'pedalboard', 'unidecode', 'onnx'],
    'heartmula': ['transformers==4.57.0', 'tokenizers==0.22.1', 'torchtune==0.4.0', 'torchao==0.9.0',
                 'vector-quantize-pytorch==1.27.15', 'einops==0.8.1', 'soundfile'],
    'songbloom': ['transformers==4.44.1', 'tokenizers==0.19.1', 'omegaconf==2.3.0', 'spacy==3.7.5',
                 'wordsegment', 'g2p-en', 'num2words', 'jieba', 'jieba-fast', 'pypinyin', 'cn2an',
                 'descript-audio-codec==1.0.0', 'vector-quantize-pytorch==1.14.8', 'einops-exts', 'local-attention'],
}


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--model', choices=PACKAGES, required=True); a = p.parse_args()
    destination = a.root/'envs'/a.model
    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(destination)
    python = destination/'bin/python'
    lib = subprocess.check_output([str(python), '-c', 'import sysconfig; print(sysconfig.get_path("purelib"))'], text=True).strip()
    base_libs = [s for s in sys.path if s.endswith('site-packages')]
    Path(lib, 'shared-metrics.pth').write_text('\n'.join(base_libs)+'\n')
    env = {**os.environ, 'PIP_CACHE_DIR': str(a.root/'pip-cache')}
    subprocess.run([str(python), '-m', 'pip', 'install', *PACKAGES[a.model]], env=env, check=True)
    (destination/'ready.json').write_text(json.dumps({'model': a.model, 'base_python': sys.executable,
        'requested_packages': PACKAGES[a.model], 'status': 'packages-installed-not-a-GPU-training-result'})+'\n')


if __name__ == '__main__': main()
