"""Sealed all-prompt short-clip WSB test after selecting one checkpoint on CMI validation."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from peft import PeftModel

from yue2 import YuE2Pipeline
from yue2_songeval_longrun import sample, score_many, stats, write_json


WSB_SHA256 = "859e5225d319912efb64d587d1d5eb6cf0abc9fb162cc3d5f02d1688cff2edc0"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage(pipe, args, prompts, step):
    root = args.output / f"step_{step:06d}"
    if (root / "receipt.json").is_file():
        receipt = json.loads((root / "receipt.json").read_text())
        if receipt.get("n") != len(prompts):
            raise ValueError("Existing test stage has wrong prompt count")
        return receipt
    pending = []
    entries = []
    for row in prompts:
        index = row["prompt_index"]
        seed = row["candidate_ar_seeds"][0]
        prompt = {"style": row["style"], "lyrics": row["lyrics"], "cot": "off", "cfg_scale": 1.0}
        folder = root / f"prompt_{index:03d}"
        wav = folder / "audio.wav"
        rollout_path = folder / "rollout.json"
        if wav.exists() and rollout_path.exists():
            rollout = json.loads(rollout_path.read_text())
        else:
            rollout = sample(pipe, prompt, seed, wav, args.max_tokens)
            write_json(rollout_path, {"seconds": rollout["seconds"], "truncated": rollout["truncated"]})
        pending.append((wav, folder / "reward"))
        entries.append({"prompt_index": index, "seed": seed,
                        "seconds": rollout["seconds"], "truncated": rollout["truncated"]})
        if (index + 1) % 24 == 0:
            print(f"stage={step} generated={index + 1}/{len(prompts)}", flush=True)
    scores = score_many(pipe, args, pending)
    for entry, (wav, _), score in zip(entries, pending, scores):
        canonical = wav.with_suffix(".flac")
        scored = canonical if args.reward_backend == "musecritic" else wav
        entry["scored_audio"] = str(scored.relative_to(args.output))
        entry["scored_audio_sha256"] = sha256(scored)
        if args.reward_backend == "songeval":
            audio, rate = sf.read(wav, dtype="float32")
            sf.write(canonical, audio, rate, subtype="PCM_24")
        entry.update({"audio": str(canonical.relative_to(args.output)),
                      "audio_sha256": sha256(canonical), "reward": score,
                      "signal": stats(canonical)})
    receipt = {"step": step, "n": len(entries), "heldout": entries,
               "mean_reward": float(np.mean([entry["reward"]["mean"] for entry in entries]))}
    write_json(root / "receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--wsb-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reward-backend", choices=("songeval", "musecritic"), required=True)
    parser.add_argument("--songeval", type=Path)
    parser.add_argument("--scorer-python", type=Path)
    parser.add_argument("--musecritic-python", type=Path)
    parser.add_argument("--musecritic-repo", type=Path)
    parser.add_argument("--musecritic-model", type=Path)
    parser.add_argument("--musecritic-max-new-tokens", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--smoke-limit", type=int, help="Separate wiring test only, never a final benchmark")
    args = parser.parse_args()
    if sha256(args.wsb_manifest) != WSB_SHA256:
        parser.error("WildSongBench manifest SHA256 differs from sealed revision")
    if args.reward_backend == "songeval" and (not args.songeval or not args.scorer_python):
        parser.error("SongEval requires --songeval and --scorer-python")
    if args.reward_backend == "musecritic" and any(value is None for value in
            (args.musecritic_python, args.musecritic_repo, args.musecritic_model)):
        parser.error("MuseCritic requires scorer Python, repo and model")
    with args.wsb_manifest.open(encoding="utf-8") as source:
        prompts = [json.loads(line) for line in source if line.strip()]
    if len(prompts) != 192 or [row["prompt_index"] for row in prompts] != list(range(192)):
        parser.error("Expected all 192 ordered WildSongBench prompts")
    if args.smoke_limit:
        prompts = prompts[:args.smoke_limit]
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    # score_many uses this marker to batch SongEval inputs; it is not training data.
    args.dataset_manifest = args.wsb_manifest
    adapter_weights = args.adapter / "adapter_model.safetensors"
    if not adapter_weights.is_file():
        parser.error(f"Missing selected adapter: {adapter_weights}")
    manifest = {"protocol": "WSB-192-short-clip-matched-v1", "smoke_only": bool(args.smoke_limit),
                "prompt_count": len(prompts), "wsb_sha256": WSB_SHA256,
                "selected_adapter_sha256": sha256(adapter_weights),
                "reward_backend": args.reward_backend, "max_tokens": args.max_tokens,
                "seed_rule": "candidate_ar_seeds[0] from each official manifest row"}
    config_path = args.output / "experiment.json"
    if config_path.is_file() and json.loads(config_path.read_text()) != manifest:
        raise ValueError("Existing test output uses a different configuration")
    if not config_path.is_file():
        write_json(config_path, manifest)
    pipe = YuE2Pipeline.from_pretrained(args.model, vae=args.vae, device="cuda",
                                        backend="torch-eager", memory_budget_gib=32,
                                        progress=False)
    base = pipe._load_model().eval()
    before = stage(pipe, args, prompts, 0)
    PeftModel.from_pretrained(base, args.adapter, is_trainable=False).eval()
    after = stage(pipe, args, prompts, 100)
    print(f"WSB short-clip {len(prompts)} prompts: {before['mean_reward']:.4f} -> "
          f"{after['mean_reward']:.4f} on {args.reward_backend} scale", flush=True)


if __name__ == "__main__":
    main()
