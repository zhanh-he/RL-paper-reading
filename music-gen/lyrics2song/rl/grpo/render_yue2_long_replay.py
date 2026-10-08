"""Render paired long-form YuE2 validation audio without changing training receipts."""

import argparse
import hashlib
import json
from pathlib import Path

import soundfile as sf
import torch
from peft import PeftModel
from yue2 import YuE2Pipeline

from yue2_songeval_longrun import write_json
from yue2_songeval_pilot import stats


DATASET_SHA256 = "c39d43e81793bee3f312c7028a0483b5da41068db4544acd258115043101e46f"
INDEXES = (0, 10, 20)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render(pipe, prompt, seed, folder, max_tokens, min_tokens):
    wav = folder / "audio.wav"
    flac = folder / "audio.flac"
    receipt = folder / "receipt.json"
    if receipt.exists():
        row = json.loads(receipt.read_text())
        if row["seed"] != seed or row["max_tokens"] != max_tokens or row["min_tokens"] != min_tokens:
            raise ValueError(f"Changed generation configuration: {receipt}")
        if sha256(wav) != row["wav_sha256"] or sha256(flac) != row["flac_sha256"]:
            raise ValueError(f"Audio hash mismatch: {receipt}")
        return row
    if wav.exists() or flac.exists():
        raise FileExistsError(f"Incomplete prior render at {folder}")
    folder.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        song = pipe(**{**prompt, "seed": seed},
                    semantic_sampling={"max_tokens": max_tokens, "min_tokens": min_tokens})
    sf.write(wav, song.audio, song.sample_rate, subtype="FLOAT")
    sf.write(flac, song.audio, song.sample_rate, subtype="PCM_24")
    row = {
        "seed": seed, "max_tokens": max_tokens, "min_tokens": min_tokens,
        "seconds": len(song.audio) / song.sample_rate,
        "semantic_tokens": len(song.semantic.tokens), "truncated": song.truncated,
        "wav_sha256": sha256(wav), "flac_sha256": sha256(flac), "signal": stats(flac),
        "reaches_30_seconds": len(song.audio) / song.sample_rate >= 30,
        "reaches_60_seconds": len(song.audio) / song.sample_rate >= 60,
    }
    write_json(receipt, row)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--step", type=int, default=270)
    parser.add_argument("--max-tokens", type=int, default=1600)
    parser.add_argument("--min-tokens", type=int, default=200)
    args = parser.parse_args()
    if args.min_tokens >= args.max_tokens:
        parser.error("min-tokens must be below max-tokens")
    if sha256(args.dataset) != DATASET_SHA256:
        raise ValueError("Frozen CMI-Pref projection changed")
    run = json.loads((args.run_root / "experiment.json").read_text())
    verified = json.loads((args.run_root / "verified_complete.json").read_text())
    if run["dataset_sha256"] != DATASET_SHA256 or run["max_tokens"] != 600:
        raise ValueError("This is not the audited short-generation formal run")
    if verified["status"] != "verified-complete":
        raise ValueError("Source training is not verified complete")
    adapter = args.run_root / "checkpoints" / f"step_{args.step:06d}" / "adapter"
    if not (adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(adapter)
    dataset = json.loads(args.dataset.read_text())
    valid = dataset["valid"]
    output = args.run_root / "long_replay" / f"step_{args.step:06d}"
    manifest = {
        "protocol": "fixed-validation-long-inference-v1", "dataset_sha256": DATASET_SHA256,
        "source_experiment_sha256": sha256(args.run_root / "experiment.json"),
        "adapter_sha256": sha256(adapter / "adapter_model.safetensors"),
        "validation_indexes": list(INDEXES), "max_tokens": args.max_tokens,
        "min_tokens": args.min_tokens, "seed_base": run["eval_seed_base"],
        "training_max_tokens": 600, "source_checkpoint": args.step,
        "reference_audio_used": False,
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError(f"Existing long replay has different configuration: {manifest_path}")
    if not manifest_path.exists():
        write_json(manifest_path, manifest)

    pipe = YuE2Pipeline.from_pretrained(args.model, vae=args.vae, device="cuda",
                                        backend="torch-eager", memory_budget_gib=32,
                                        progress=False)
    base = pipe._load_model().eval()
    for index in INDEXES:
        row = valid[index]
        prompt = {"style": row["style"], "lyrics": row["lyrics"],
                  "cot": "off", "cfg_scale": 1.0}
        result = render(pipe, prompt, run["eval_seed_base"] + index,
                        output / "baseline" / f"valid_{index:02d}", args.max_tokens, args.min_tokens)
        print(f"baseline valid_{index:02d}: {result['seconds']:.2f}s", flush=True)
    PeftModel.from_pretrained(base, adapter, is_trainable=False).eval()
    for index in INDEXES:
        row = valid[index]
        prompt = {"style": row["style"], "lyrics": row["lyrics"],
                  "cot": "off", "cfg_scale": 1.0}
        result = render(pipe, prompt, run["eval_seed_base"] + index,
                        output / "grpo" / f"valid_{index:02d}", args.max_tokens, args.min_tokens)
        print(f"grpo valid_{index:02d}: {result['seconds']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
