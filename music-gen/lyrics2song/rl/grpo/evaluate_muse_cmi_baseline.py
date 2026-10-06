"""Fixed-seed 60-condition Muse GRPO validation; no test-set selection."""

import argparse
import gc
import hashlib
import json
import math
import subprocess
from pathlib import Path

import soundfile as sf
import torch
from peft import PeftModel

from muse_songeval_pilot import load, rollout, signal


VALID_SHA256 = "3c6ad89c4841041eb64f51536e544b5a925fd450e334f7742025a90f4f5be211"
SCORE_KEYS = ("Coherence", "Musicality", "Memorability", "Clarity", "Naturalness")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prompts(path):
    if sha256(path) != VALID_SHA256:
        raise ValueError("Frozen Muse validation JSONL changed")
    records = [json.loads(line) for line in path.read_text().split("\n") if line]
    if len(records) != 60:
        raise ValueError("Expected all 60 held-out conditions")
    return [record["messages"][0]["content"] for record in records]


def score(args, files, output):
    if args.reward_backend == "songeval":
        batch = output / "songeval"
        batch.mkdir(exist_ok=True)
        input_list = batch / "inputs.txt"
        input_list.write_text("".join(f"{path.resolve()}\n" for path in files))
        subprocess.run([str(args.scorer_python), "eval.py", "-i", str(input_list),
                        "-o", str(batch / "scores")], cwd=args.songeval, check=True)
        scored = json.loads((batch / "scores" / "result.json").read_text())
        return [scored[path.stem] for path in files]

    pairs = []
    for index, wav in enumerate(files):
        audio, sample_rate = sf.read(wav, dtype="float32")
        canonical = wav.with_suffix(".flac")
        sf.write(canonical, audio, sample_rate, subtype="PCM_24")
        pairs.extend(["--pair", str(canonical), str(output / f"critic_{index:04d}")])
    subprocess.run([str(args.scorer_python), str(args.critic_script),
                    "--repo", str(args.critic_repo), "--model", str(args.critic_model),
                    "--max-new-tokens", "512", *pairs], check=True)
    return [json.loads((output / f"critic_{index:04d}" / "result.json").read_text())[
        wav.stem] for index, wav in enumerate(files)]


def evaluate(args):
    condition_prompts = prompts(args.valid_jsonl)
    if args.output.exists() and (args.output / "receipt.json").exists():
        raise FileExistsError("Refusing to overwrite verified evaluation")
    args.output.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load(args.model)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload().eval()
    files = []
    generations = []
    for index, prompt in enumerate(condition_prompts):
        path = args.output / f"heldout_{index:04d}.json"
        if path.exists():
            row = json.loads(path.read_text())
        else:
            row = rollout(model, tokenizer, prompt, args.seed_base + index,
                          args.max_tokens, path, force_length=True)
        if len(row["audio_tokens"]) < 32:
            raise ValueError(f"Insufficient Muse audio tokens for heldout {index}")
        files.append(path.with_suffix(".wav"))
        generations.append({"index": index, "seed": args.seed_base + index,
                            "generated_tokens": len(row["generated_ids"]),
                            "audio_tokens": len(row["audio_tokens"])})
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    if any(not path.is_file() for path in files):
        subprocess.run([str(args.codec_python), str(args.decoder),
                        "--mucodec", str(args.mucodec), "--steps", "20",
                        "--duration", "10.24", "--token-file",
                        *[str(path.with_suffix(".json")) for path in files if not path.is_file()]],
                       check=True, cwd=args.mucodec)
    dimensions = score(args, files, args.output)
    entries = []
    for generation, wav, values in zip(generations, files, dimensions):
        if set(values) != set(SCORE_KEYS) or not all(math.isfinite(float(value)) for value in values.values()):
            raise ValueError(f"Invalid {args.reward_backend} score for {wav}")
        canonical = wav.with_suffix(".flac")
        if args.reward_backend == "songeval":
            audio, sample_rate = sf.read(wav, dtype="float32")
            sf.write(canonical, audio, sample_rate, subtype="PCM_24")
        scored = canonical if args.reward_backend == "musecritic" else wav
        entries.append({**generation, "audio": str(canonical.relative_to(args.output)),
                        "audio_sha256": sha256(canonical),
                        "scored_audio": str(scored.relative_to(args.output)),
                        "scored_audio_sha256": sha256(scored),
                        "reward": {**values, "mean": sum(float(x) for x in values.values()) / 5},
                        "signal": signal(canonical)})
    receipt = {
        "status": "verified-fixed-validation",
        "model": "Muse-0.6b + MuCodec",
        "actual_input_modalities": ["text", "lyrics"],
        "reference_audio_used": False,
        "validation_jsonl_sha256": VALID_SHA256,
        "reward_backend": args.reward_backend,
        "adapter": str(args.adapter) if args.adapter else None,
        "seed_base": args.seed_base,
        "max_tokens": args.max_tokens,
        "force_length": True,
        "n": len(entries),
        "mean_reward": sum(item["reward"]["mean"] for item in entries) / len(entries),
        "heldout": entries,
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--mucodec", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--codec-python", type=Path, required=True)
    parser.add_argument("--scorer-python", type=Path, required=True)
    parser.add_argument("--reward-backend", choices=("songeval", "musecritic"), required=True)
    parser.add_argument("--songeval", type=Path)
    parser.add_argument("--critic-script", type=Path)
    parser.add_argument("--critic-repo", type=Path)
    parser.add_argument("--critic-model", type=Path)
    parser.add_argument("--valid-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-base", type=int, default=5101)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()
    if args.reward_backend == "songeval" and args.songeval is None:
        parser.error("SongEval requires --songeval")
    if args.reward_backend == "musecritic" and any(value is None for value in
            (args.critic_script, args.critic_repo, args.critic_model)):
        parser.error("MuseCritic requires script, repo and model")
    receipt = evaluate(args)
    print(f"{receipt['n']} fixed validation songs; {args.reward_backend} mean={receipt['mean_reward']:.4f}")


if __name__ == "__main__":
    main()
