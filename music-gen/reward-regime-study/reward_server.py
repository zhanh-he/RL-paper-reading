"""Serialized lossless-audio scorer; models return to CPU between requests."""
import argparse
import gc
import json
import sys
import traceback
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np
import soundfile as sf
import torch
from rewards import sha, guard, spectral_richness, objective


class Scorer:
    def __init__(self, args):
        sys.path[:0] = [str(args.deps), str(args.bundle/'src'), str(args.bundle/'vendor')]
        from mot_infer.evaluation import SongBench
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
        self.apply = apply_model
        self.bench = SongBench(args.bundle, device='cpu')
        original = torch.load
        def trusted_load(*a, **kw):
            kw.setdefault('weights_only', False)
            return original(*a, **kw)
        self.identity = {'reward_code_sha256': sha(Path(__file__).with_name('rewards.py')),
                         'server_sha256': sha(Path(__file__)), 'separator': 'htdemucs-955717e8',
                         'separator_sha256': sha(args.separator),
                         'songbench_weights_sha256': sha(args.bundle/'models/songbench/songbench.safetensors'),
                         'torch': torch.__version__, 'scored_format': 'PCM24-FLAC',
                         'separator_shifts': 0, 'separator_overlap': .25,
                         'songbench_input': 'whole mix', 'richness_input': 'estimated accompaniment'}
        try:
            torch.load = trusted_load
            self.sep = get_model('955717e8', repo=args.separator.parent).eval().cpu()
        finally:
            torch.load = original
        self.output = args.receipt_dir
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output/'scorer-identity.json').write_text(json.dumps(self.identity, indent=2)+'\n')

    def score(self, paths, reward):
        from torchaudio.functional import resample
        rows = []
        try:
            for name in paths:
                path = Path(name)
                if not path.is_absolute() or path.suffix != '.flac':
                    raise ValueError('Scorer requires absolute canonical lossless FLAC')
                audio, sr = sf.read(path, dtype='float32', always_2d=True)
                signal = guard(audio, sr)
                if signal['invalid']:
                    rich = {'raw': 0.}; dimensions = {key: 0. for key in ['Melody','Arrangement','Musicality','Vocal','Instrumental','Mixing','Structure']}
                else:
                    self.sep.to('cuda')
                    wave = torch.from_numpy(audio.T.copy())
                    if wave.shape[0] == 1: wave = wave.repeat(2, 1)
                    wave = resample(wave, sr, self.sep.samplerate).to('cuda')
                    ref = wave.mean(0); mean = ref.mean(); std = ref.std().clamp_min(1e-6)
                    with torch.inference_mode():
                        stems = self.apply(self.sep, ((wave-mean)/std)[None], device='cuda',
                            shifts=0, split=True, overlap=.25, progress=False)[0]*std+mean
                        accompaniment = stems[[i for i,n in enumerate(self.sep.sources) if n != 'vocals']].sum(0)
                    rich = spectral_richness(accompaniment.cpu().numpy().T, self.sep.samplerate)
                    del stems, accompaniment, wave, ref, mean, std
                    self.sep.cpu(); torch.cuda.empty_cache()
                    self.bench.muq.to('cuda'); self.bench.reward.to('cuda'); self.bench.device = torch.device('cuda')
                    torch.manual_seed(42)
                    dimensions = self.bench.score(path)['songbench_raw']
                    self.bench.muq.cpu(); self.bench.reward.cpu(); self.bench.device = torch.device('cpu')
                    torch.cuda.empty_cache()
                result = {'audio_sha256': sha(path), 'seconds': len(audio)/sr,
                          'richness': rich, 'songbench': dimensions, 'guard': signal,
                          'reward': reward, 'scorer_identity': self.identity}
                result['mean'] = objective(result, reward)
                (self.output/f'{result["audio_sha256"]}-{reward}.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
                rows.append(result)
        finally:
            self.sep.cpu(); self.bench.muq.cpu(); self.bench.reward.cpu()
            self.bench.device = torch.device('cpu'); gc.collect(); torch.cuda.empty_cache()
        return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True); p.add_argument('--deps', type=Path, required=True)
    p.add_argument('--separator', type=Path, required=True); p.add_argument('--receipt-dir', type=Path, required=True)
    p.add_argument('--port', type=int, required=True); p.add_argument('--container-work', type=Path)
    args = p.parse_args()
    if args.container_work:
        from enter_eval_container import enter
        enter(args.container_work)
    scorer = Scorer(args)
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200); self.end_headers(); self.wfile.write(b'{"ready":true}')
        def do_POST(self):
            try:
                data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if data['reward'] not in ['richness','instrument','arrangement']: raise ValueError('Unknown reward')
                result = {'rows': scorer.score(data['paths'], data['reward'])}; status = 200
            except Exception as error:
                traceback.print_exc(); result = {'error': str(error)}; status = 500
            self.send_response(status); self.end_headers()
            self.wfile.write(json.dumps(result, allow_nan=False).encode())
    HTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__': main()
