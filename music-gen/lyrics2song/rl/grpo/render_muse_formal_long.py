"""Render fixed CMI-Pref Muse replays with official overlapping MuCodec decoding."""

import argparse
import gc
import hashlib
import json
import subprocess
from pathlib import Path

import soundfile as sf
import torch
from peft import PeftModel

from muse_songeval_pilot import load, rollout, signal


VALID_SHA = 'ed5c5f36af48ef6ce81d15c7b15b429fb250ec4a52a04a43e55d96ec3cafb59d'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'valid', 'output', 'mucodec', 'codec-python', 'decoder'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--adapter', type=Path)
    args = parser.parse_args()
    if digest(args.valid) != VALID_SHA:
        raise ValueError('Frozen validation projection changed')
    args.output.mkdir(parents=True, exist_ok=True)
    prompts = [json.loads(line)['messages'][0]['content'] for line in args.valid.read_text().splitlines()]
    pending = []
    for index in (0, 10, 20):
        folder = args.output / f'valid_{index:02d}'
        receipt = folder / 'receipt.json'
        if receipt.exists():
            existing = json.loads(receipt.read_text())
            if digest(folder / 'audio.flac') != existing['flac_sha256']:
                raise ValueError(f'Changed published audio: {folder}')
            continue
        folder.mkdir(exist_ok=True)
        pending.append((index, folder))
    if not pending:
        return
    model, tokenizer = load(args.model)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload().eval()
    for index, folder in pending:
        tokens = folder / 'audio.json'
        if not tokens.exists():
            rollout(model, tokenizer, prompts[index], 5101 + index, 1600, tokens, force_length=True)
        row = json.loads(tokens.read_text())
        if row['seed'] != 5101 + index or row['max_new_tokens'] != 1600 or not row['force_length']:
            raise ValueError(f'Generation settings mismatch: {tokens}')
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    files = [folder / 'audio.json' for _, folder in pending if not (folder / 'audio.wav').exists()]
    if files:
        subprocess.run([str(args.codec_python), str(args.decoder), '--mucodec', str(args.mucodec),
                        '--duration', '40.96', '--steps', '20', '--token-file', *map(str, files)],
                       check=True, cwd=args.mucodec)
    for index, folder in pending:
        audio, rate = sf.read(folder / 'audio.wav', dtype='float32')
        if rate != 48000:
            raise ValueError('Unexpected MuCodec sample rate')
        sf.write(folder / 'audio.flac', audio, rate, subtype='PCM_24')
        generated = json.loads((folder / 'audio.json').read_text())
        result = {
            'index': index, 'seed': 5101 + index, 'max_tokens': 1600, 'force_length': True,
            'audio_tokens': len(generated['audio_tokens']), 'seconds': len(audio) / rate,
            'decoder_window_seconds': 40.96, 'decoder_overlap_native': True,
            'wav_sha256': digest(folder / 'audio.wav'), 'flac_sha256': digest(folder / 'audio.flac'),
            'signal': signal(folder / 'audio.flac'), 'truncated': True,
            'adapter_sha256': digest(args.adapter / 'adapter_model.safetensors') if args.adapter else None,
            'validation_jsonl_sha256': VALID_SHA,
            'reaches_30_seconds': len(audio) / rate >= 30,
        }
        (folder / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n')
        print(f'Muse valid_{index:02d}: {result["seconds"]:.2f}s', flush=True)


if __name__ == '__main__':
    main()
