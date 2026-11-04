"""Score fixed excerpts using official SongEval head and pinned MuQ on CPU."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import sys
import time
import librosa
import numpy as np
import torch
import yaml
from muq import MuQ
from safetensors.torch import load_file

LABELS = ['Coherence','Musicality','Memorability','Clarity','Naturalness']

def sha(path):
    digest = hashlib.sha256()
    with open(path,'rb') as source:
        while block := source.read(8<<20):
            digest.update(block)
    return digest.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--audio-dir',required=True)
    parser.add_argument('--songeval',required=True)
    parser.add_argument('--muq',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--threads',type=int,default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    root = Path(args.songeval)
    sys.path.insert(0,str(root.resolve()))
    from model import Generator
    config = yaml.safe_load((root/'config.yaml').read_text())['generator']
    assert config.pop('_target_') == 'model.Generator'
    began = time.perf_counter()
    head = Generator(**config).to('cpu').eval()
    loaded = head.load_state_dict(load_file(root/'ckpt/model.safetensors'),strict=True)
    backbone = MuQ.from_pretrained(args.muq).to('cpu').eval()
    initialization_seconds = time.perf_counter()-began
    manifest = json.loads(Path(args.manifest).read_text())
    rows = []
    for record in manifest['samples']:
        path = Path(args.audio_dir)/(record['id']+'.wav')
        audio_hash = sha(path)
        assert audio_hash == record['excerpt_sha256']
        start = time.perf_counter()
        # Identical to official eval.py: librosa mono with sr=24000, hidden[6].
        audio,sr = librosa.load(path,sr=24000,mono=True)
        assert sr == 24000 and np.isfinite(audio).all() and len(audio)==240000
        with torch.inference_mode():
            features = backbone(torch.from_numpy(audio).unsqueeze(0),output_hidden_states=True)['hidden_states'][6]
            values = head(features).squeeze(0).float().cpu().tolist()
        assert len(values)==5 and np.isfinite(values).all()
        row = {'id':record['id'],'audio_sha256':audio_hash,
               'scores':dict(zip(LABELS,values)),
               'segment_spans':[{'start':0.,'end':10.}],
               'inference_seconds':time.perf_counter()-start}
        rows.append(row)
        print(json.dumps(row),flush=True)
    times = np.array([r['inference_seconds'] for r in rows])
    metadata = {'method':'SongEval','device':'cpu','hardware':'Existing Linux x86_64 CPU host',
                'threads':args.threads,'code_commit':'848fb2ff3a2a9d64dcb20d46f07238c28abd7add',
                'official_code':'https://github.com/ASLP-lab/SongEval',
                'head_sha256':sha(root/'ckpt/model.safetensors'),
                'config_sha256':sha(root/'config.yaml'),
                'head_code_sha256':sha(root/'model.py'),
                'backbone_hf_repo':'OpenMuQ/MuQ-large-msd-iter',
                'backbone_revision':Path(args.muq).name,
                'backbone_sha256':sha(Path(args.muq)/'model.safetensors'),
                'model_init_seconds':initialization_seconds,
                'total_seconds_including_init':time.perf_counter()-began,
                'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                'preprocessing':'Official librosa mono24kHz with default soxr_hq resampling; MuQ hidden_states[6]; one10s segment; no gain normalization.',
                'timing_scope':'Sequential25 clips, CPU4threads. Includes librosa decode/resample and model forward; init excludes Python imports. Hardware differs from local Audiobox/PAM benchmarks; speed is not a matched leaderboard.',
                'mean_seconds_per_10s_clip':float(times.mean()),
                'median_seconds_per_10s_clip':float(np.median(times)),
                'min_seconds':float(times.min()),'max_seconds':float(times.max()),
                'model_load_key_receipt':{'missing_keys':loaded.missing_keys,'unexpected_keys':loaded.unexpected_keys},
                'human_labels_count':0,
                'packages':{name:importlib.metadata.version(name) for name in ['torch','muq','transformers','librosa','numpy','safetensors','einops','PyYAML']}}
    Path(args.output).write_text(json.dumps({'schema_version':1,'method':metadata,'samples':rows},indent=2)+'\n')

if __name__=='__main__':
    main()
