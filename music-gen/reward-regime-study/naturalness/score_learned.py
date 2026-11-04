"""Local, traceable Audiobox/PAM exploratory scoring (no perceptual labels)."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np
import soundfile as sf
import torch
import torchaudio

ARTIFACT_PROMPTS = [
    'The music is clear and natural without unwanted digital artifacts.',
    'The music has unnatural metallic buzzing and digital artifacts.',
]

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        while block := stream.read(8 << 20):
            h.update(block)
    return h.hexdigest()


def load_samples(path):
    data = json.loads(Path(path).read_text())
    return data if isinstance(data, list) else data['samples']


def audio_path(row):
    return row.get('path') or row.get('scoring_path') or row.get('local_path')


def read_mono(path):
    audio, sr = sf.read(path, dtype='float32', always_2d=True)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError('Empty or nonfinite audio')
    stats = {'float_peak': float(np.max(np.abs(audio))),
             'outside_unit_fraction': float(np.mean(np.abs(audio) > 1)),
             'channels': audio.shape[1], 'external_gain_normalization': False}
    return torch.from_numpy(audio.mean(axis=1).copy()).unsqueeze(0), sr, stats


def sync(device):
    if str(device) == 'mps':
        torch.mps.synchronize()


def init_aes(args):
    from audiobox_aesthetics.infer import initialize_predictor
    old = torch.backends.mps.is_available
    if args.device == 'cpu':
        torch.backends.mps.is_available = lambda: False
    try:
        predictor = initialize_predictor(str(Path(args.aes_checkpoint).resolve()))
    finally:
        torch.backends.mps.is_available = old
    return predictor


def init_pam(args):
    # The complete CLAP checkpoint contains GPT-2 weights. Build the same GPT-2
    # architecture from the downloaded official config, then require checkpoint
    # key coverage. This avoids downloading redundant standalone GPT-2 weights.
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    sys.path.insert(0, str(Path(args.pam_source).resolve()))
    from PAM import PAM
    original_model = AutoModel.from_pretrained
    original_tokenizer = AutoTokenizer.from_pretrained
    original_load = torch.nn.Module.load_state_dict
    load_receipts = []

    def from_config(*unused_args, **unused_kwargs):
        return AutoModel.from_config(AutoConfig.from_pretrained(args.gpt2_dir))

    def tokenizer(*unused_args, **unused_kwargs):
        return original_tokenizer(args.gpt2_dir)

    def checked_load(model, state_dict, *posargs, **kwargs):
        result = original_load(model, state_dict, *posargs, **kwargs)
        load_receipts.append({'missing_keys': result.missing_keys,
                              'unexpected_keys': result.unexpected_keys})
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(f'CLAP checkpoint mismatch: {load_receipts[-1]}')
        return result

    AutoModel.from_pretrained = from_config
    AutoTokenizer.from_pretrained = tokenizer
    torch.nn.Module.load_state_dict = checked_load
    try:
        predictor = PAM(model_fp=str(Path(args.pam_checkpoint).resolve()), use_cuda=False)
    finally:
        AutoModel.from_pretrained = original_model
        AutoTokenizer.from_pretrained = original_tokenizer
        torch.nn.Module.load_state_dict = original_load
    predictor.artifact_embeddings = predictor._get_text_embeddings(
        predictor.preprocess_text(ARTIFACT_PROMPTS))
    return predictor, load_receipts


def pam_chunks(wav, sr):
    wav = torchaudio.functional.resample(wav, sr, 44100).squeeze(0)
    length = 7 * 44100
    if len(wav) <= length:
        wav = wav.repeat(int(np.ceil(length / len(wav))))[:length]
        return wav.unsqueeze(0), [{'start': 0., 'end': 7., 'short_clip_repeated': True}]
    count = len(wav) // length
    chunks = [wav[i*length:(i+1)*length] for i in range(count)]
    spans = [{'start': 7.*i, 'end': 7.*(i+1)} for i in range(count)]
    if len(wav) % length:
        chunks.append(wav[-length:])
        spans.append({'start': len(wav)/44100-7, 'end': len(wav)/44100})
    return torch.stack(chunks), spans


def pam_score(predictor, wav, sr):
    chunks, spans = pam_chunks(wav, sr)
    with torch.inference_mode():
        embeddings = predictor._get_audio_embeddings(chunks)
        official = torch.softmax(predictor.compute_similarity(embeddings), dim=1)[:, 0]
        # Explicitly exploratory prompt probe; this is not the validated PAM score.
        text = predictor.artifact_embeddings
        a = torch.nn.functional.normalize(embeddings, dim=-1)
        t = torch.nn.functional.normalize(text, dim=-1)
        logits = predictor.clap.logit_scale.exp() * (a @ t.T)
        probe = torch.softmax(logits, dim=1)[:, 0]
    return {'PAM': float(official.mean()), 'PAM_segments': official.tolist(),
            'artifact_prompt_probe': float(probe.mean()),
            'artifact_prompt_segments': probe.tolist(), 'segment_spans': spans,
            'artifact_prompt_pair': ARTIFACT_PROMPTS}, embeddings.detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--method', choices=['audiobox', 'pam'], required=True)
    parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    parser.add_argument('--aes-checkpoint', default='work/model-cache/audiobox-checkpoint.pt')
    parser.add_argument('--pam-checkpoint', default='work/model-cache/msclap/CLAP_weights_2023.pth')
    parser.add_argument('--pam-source', default='work/research-upstream/PAM')
    parser.add_argument('--gpt2-dir', default='work/model-cache/gpt2')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--embedding-dir', default='work/embeddings/pam')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.method == 'pam' and args.device != 'cpu':
        parser.error('Official PAM wrapper in this experiment uses CPU')
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    samples = load_samples(args.manifest)
    if args.limit:
        samples = samples[:args.limit]
    began = time.perf_counter()
    receipts = None
    if args.method == 'audiobox':
        predictor = init_aes(args)
        checkpoint = args.aes_checkpoint
        device = str(predictor.device)
    else:
        predictor, receipts = init_pam(args)
        checkpoint = args.pam_checkpoint
        device = 'cpu'
    init_seconds = time.perf_counter() - began
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in samples:
        path = audio_path(row)
        wav, sr, input_stats = read_mono(path)
        sync(device)
        start = time.perf_counter()
        try:
            if args.method == 'audiobox':
                score = predictor.forward([{'path': wav, 'sample_rate': sr}])[0]
            else:
                score, embeddings = pam_score(predictor, wav, sr)
                embedding_path = Path(args.embedding_dir) / (row['id']+'.npy')
                embedding_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(embedding_path, embeddings)
            sync(device)
            result = {'id': row['id'], 'audio_sha256': sha(path), 'sample_rate': sr,
                      'duration': wav.shape[-1] / sr, 'scores': score,
                      'input_stats': input_stats,
                      'inference_seconds': time.perf_counter()-start}
            if args.method == 'pam':
                result['embedding_sha256'] = sha(embedding_path)
        except Exception as error:
            result = {'id': row['id'], 'audio_sha256': sha(path),
                      'error': repr(error), 'inference_seconds': time.perf_counter()-start}
        rows.append(result)
        with destination.open('a') as stream:
            stream.write(json.dumps(result) + '\n')
        print(json.dumps(result), flush=True)
    metadata = {'method': args.method, 'device': device, 'threads': args.threads,
                'hardware': 'Apple M4 Mac mini, 24 GiB unified memory, macOS arm64',
                'python': platform.python_version(), 'torch': torch.__version__,
                'checkpoint_sha256': sha(checkpoint),
                'checkpoint_bytes': Path(checkpoint).stat().st_size,
                'model_init_seconds': init_seconds,
                'total_seconds_including_init': time.perf_counter()-began,
                'peak_process_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'count': len(rows), 'manifest_sha256': sha(args.manifest),
                'model_load_key_receipts': receipts,
                'packages': {x:importlib.metadata.version(x) for x in
                             ['numpy','torch','torchaudio','audiobox_aesthetics','transformers',
                              'huggingface_hub','soundfile','scipy']}}
    destination.with_suffix('.metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')


if __name__ == '__main__':
    main()
