"""Replay saved LaDA adapters on a vocal phrase excluded from online updates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf
import torch

from train_lada_band import OutputLoRA, decode, encode_condition, frame_rms, sample_trajectory, score_audio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-seconds", type=float, default=6)
    parser.add_argument("--duration-seconds", type=float, default=6)
    parser.add_argument("--steps", type=int, nargs="+", default=[0, 5, 50, 100, 300])
    args = parser.parse_args()

    run_args = json.loads((args.run_dir / "run.json").read_text())["args"]
    cfg = argparse.Namespace(**run_args)
    code_root = Path(cfg.code_root).resolve()
    sys.path.insert(0, str(code_root))
    from omegaconf import OmegaConf
    from lada_band.module.llada import ModelModule
    from lada_band.utils.config_utils import resolve_relative_paths

    vocal_path = Path(cfg.vocal)
    info = sf.info(vocal_path)
    if info.samplerate != 48000:
        raise ValueError("this fixed-vocal replay expects 48 kHz PCM")
    start = round(args.start_seconds * info.samplerate)
    frames = round(args.duration_seconds * info.samplerate)
    audio, _ = sf.read(vocal_path, start=start, frames=frames, dtype="float32", always_2d=True)
    if len(audio) != frames:
        raise ValueError("requested held-out phrase exceeds the source vocal")
    vocal = torch.from_numpy(audio.T.copy()).cuda()
    if vocal.shape[0] == 1:
        vocal = vocal.repeat(2, 1)
    vocal_rms = frame_rms(audio.mean(axis=1), info.samplerate)

    model_config = code_root / "lada_band/conf/infer_1B.yaml"
    config_dict = OmegaConf.to_container(OmegaConf.load(model_config), resolve=True)
    model_cfg = OmegaConf.create(resolve_relative_paths(config_dict, str(model_config)))
    torch.manual_seed(cfg.seed)
    module = ModelModule(model_cfg)
    module.load_model(cfg.checkpoint)
    module = module.cuda().eval()
    module.requires_grad_(False)
    model = module.model
    adapter = OutputLoRA(model.to_logits, cfg.rank, cfg.alpha).cuda()
    model.to_logits = adapter
    voc_ids, condition = encode_condition(module, vocal, cfg.text)

    args.output.mkdir(parents=True, exist_ok=True)
    sf.write(args.output / "source.wav", audio, info.samplerate)
    records = []
    for step in args.steps:
        if step == 0:
            adapter.b.weight.data.zero_()
        else:
            checkpoint = torch.load(args.run_dir / f"step_{step:04d}.pt", map_location="cpu", weights_only=False)
            if int(checkpoint["step"]) != step:
                raise ValueError(f"checkpoint step mismatch: {step}")
            adapter.a.load_state_dict(checkpoint["a"])
            adapter.b.load_state_dict(checkpoint["b"])
        tokens, _ = sample_trajectory(model, voc_ids, condition, cfg, cfg.eval_seed)
        generated = decode(module, tokens)
        sf.write(args.output / f"step_{step:04d}.wav", generated, 48000)
        record = {
            "step": step,
            "source_start_seconds": args.start_seconds,
            "duration_seconds": args.duration_seconds,
            **score_audio(generated, vocal_rms, 48000, cfg.reward),
        }
        records.append(record)
        print(json.dumps(record), flush=True)
    (args.output / "evaluations.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
