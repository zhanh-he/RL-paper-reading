"""Supplementary SongEval / AudioBox scores on the same selected scoring MP3s."""

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

from generate_objective import sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--chunk', type=Path, required=True)
    p.add_argument('--songeval', type=Path, required=True)
    a = p.parse_args()
    import torch
    import librosa
    import numpy as np
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from safetensors.torch import load_file
    from muq import MuQ
    from audiobox_aesthetics.infer import initialize_predictor
    sys.path.insert(0, str(a.songeval))
    config = OmegaConf.load(a.songeval / 'config.yaml')
    head = instantiate(config.generator).cuda().eval()
    checkpoint = a.songeval / 'ckpt/model.safetensors'
    if not checkpoint.exists():
        files = list((a.songeval / 'ckpt').glob('*.safetensors'))
        if len(files) != 1:
            raise ValueError('SongEval scorer checkpoint is ambiguous')
        checkpoint = files[0]
    head.load_state_dict(load_file(str(checkpoint)), strict=True)
    muq = MuQ.from_pretrained('OpenMuQ/MuQ-large-msd-iter').cuda().eval()
    muq_state = hashlib.sha256()
    for name, tensor in sorted(muq.state_dict().items()):
        muq_state.update(name.encode())
        muq_state.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    scorer = {'songeval_head_sha256': sha(checkpoint), 'songeval_config_sha256': sha(a.songeval / 'config.yaml'),
              'muq_state_sha256': muq_state.hexdigest(), 'scorer_code_sha256': sha(Path(__file__)),
              'audiobox_version': importlib.metadata.version('audiobox-aesthetics'),
              'scope': 'supplementary local implementation; not pinned WSB paper acceptance'}
    data = json.loads(a.chunk.read_text())
    keys = ['Coherence', 'Musicality', 'Memorability', 'Clarity', 'Naturalness']
    for row in data['rows']:
        if sha(row['audio_path']) != row['audio_sha256']:
            raise ValueError('Scoring audio hash changed')
        wav, _ = librosa.load(row['audio_path'], sr=24000, mono=True)
        if not len(wav) or not np.isfinite(wav).all():
            raise ValueError('Invalid audio')
        with torch.inference_mode():
            features = muq(torch.from_numpy(wav).unsqueeze(0).cuda(), output_hidden_states=True)['hidden_states'][6]
            scores = head(features).squeeze(0).cpu().tolist()
        if len(scores) != 5 or not np.isfinite(scores).all():
            raise ValueError('Invalid SongEval scores')
        row['supplementary'] = {'songeval': dict(zip(keys, scores)), 'scorer': scorer}
        print(f'SongEval {row["id"]}: {sum(scores)/5:.4f}', flush=True)
    del head, muq
    torch.cuda.empty_cache()
    predictor = initialize_predictor()
    aes_state = hashlib.sha256()
    for name, tensor in sorted(predictor.model.state_dict().items()):
        aes_state.update(name.encode())
        aes_state.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    for row in data['rows']:
        # Passing decoded samples avoids implicit backend changes in torchaudio.
        wav, sr = librosa.load(row['audio_path'], sr=None, mono=True)
        scores = predictor.forward([{'path': torch.from_numpy(wav).unsqueeze(0), 'sample_rate': sr}])[0]
        if not all(np.isfinite(v) for v in scores.values()):
            raise ValueError('Invalid AudioBox scores')
        row['supplementary'].update(audiobox=scores, audiobox_state_sha256=aes_state.hexdigest())
        print(f'AudioBox {row["id"]}: PQ {scores["PQ"]:.4f}', flush=True)
    write(a.chunk, data)


if __name__ == '__main__':
    main()
