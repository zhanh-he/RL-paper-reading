"""Decode Muse AUDIO tokens with the official MuCodec implementation."""

import argparse
import json
import sys
from pathlib import Path

import torch
import torchaudio

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mucodec", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    sys.path.insert(0, str(args.mucodec.resolve()))
    from generate import MuCodec

    decoder = MuCodec(str(args.mucodec / "ckpt/mucodec.pt"), layer_num=7, load_main_model=True)
    for path in args.token_file:
        row = json.loads(path.read_text())
        tokens = row["audio_tokens"]
        if len(tokens) < 32:
            raise ValueError(f"Too few audio tokens in {path}: {len(tokens)}")
        torch.manual_seed(20260929)
        torch.cuda.manual_seed_all(20260929)
        codes = torch.tensor(tokens, dtype=torch.long).view(1, 1, -1)
        with torch.inference_mode():
            wave = decoder.code2sound(codes, prompt=None, duration=40.96,
                                      guidance_scale=1.5, num_steps=args.steps,
                                      disable_progress=True)
        output = path.with_suffix(".wav")
        torchaudio.save(str(output), wave.detach().float().cpu(), 48000)
        print(f"decoded {path.name}: {len(tokens)} tokens -> {output.name}", flush=True)


if __name__ == "__main__":
    main()
