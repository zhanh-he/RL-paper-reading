"""Resume a small-prompt on-policy YuE2 SongEval-GRPO run from the one-step LoRA.

This experiment uses optimizer updates as its step count. It saves fixed-seed
held-out audio and adapters at milestones; it does not assign quality labels to
steps before listening and measuring them.
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from peft import PeftModel

from yue2 import YuE2Pipeline
from yue2_songeval_pilot import action_log_probs, score_file, stats


def request(style, line1, line2):
    return {"style": style, "lyrics": f"[Verse]\n{line1}\n{line2}",
            "cot": "off", "cfg_scale": 1.0}


TRAIN = [
    request("English, bright indie pop, piano, bass, drums, female vocal, 105 BPM",
            "Morning paints the window gold", "We let the quiet story unfold"),
    request("English, warm acoustic pop, guitar, bass, light drums, male vocal, 98 BPM",
            "The river carries yesterday", "And leaves a clearer sky today"),
    request("English, reflective folk, fingerpicked guitar, soft percussion, female vocal, 86 BPM",
            "A paper moon above the town", "Keeps every wandering thought around"),
    request("English, upbeat synth pop, pulsing bass, drums, female vocal, 118 BPM",
            "The city wakes in shades of blue", "I find a little light in you"),
    request("English, mellow soul, electric piano, bass, brushed drums, male vocal, 92 BPM",
            "The last train turns the corner slow", "I hear the midnight breezes blow"),
    request("English, gentle country pop, acoustic guitar, fiddle, drums, female vocal, 104 BPM",
            "A winding road can lead us far", "We navigate by one bright star"),
    request("English, soft rock, clean guitar, bass, drums, male vocal, 110 BPM",
            "The open door invites the sun", "Another day has just begun"),
    request("English, dreamy chamber pop, piano, strings, soft drums, female vocal, 90 BPM",
            "The evening settles on the shore", "I hear the waves and ask for more"),
]
HELDOUT = [
    request("English, mellow acoustic folk, guitar, light percussion, male vocal, 92 BPM",
            "Under open skies we roam", "Every little road leads home"),
    request("English, light jazz pop, piano, upright bass, brushes, female vocal, 100 BPM",
            "The cafe lights begin to glow", "A gentle rhythm starts to flow"),
    request("English, airy electronic pop, synth pads, soft beat, male vocal, 108 BPM",
            "Beyond the glass the colors rise", "A silver cloud across the skies"),
]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def sample(pipe, prompt, seed, wav, max_tokens):
    wav.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        song = pipe(**{**prompt, "seed": seed},
                    semantic_sampling={"max_tokens": max_tokens, "min_tokens": 200})
    sf.write(wav, song.audio, song.sample_rate, subtype="FLOAT")
    return {"prefix": song.semantic.plan.prefix, "semantic": song.semantic.tokens,
            "seconds": len(song.audio) / song.sample_rate, "truncated": song.truncated}


def evaluate(pipe, step, args):
    root = args.output / f"step_{step:06d}"
    if (root / "receipt.json").exists():
        return
    entries = []
    for index, prompt in enumerate(HELDOUT):
        folder = root / f"heldout_{index}"
        wav = folder / "audio.wav"
        rollout = sample(pipe, prompt, 5101 + index, wav, args.max_tokens)
        reward = score_file(args.songeval, args.scorer_python, wav, folder / "reward")
        audio, sr = sf.read(wav, dtype="float32")
        sf.write(folder / "audio.flac", audio, sr, subtype="PCM_24")
        signal = stats(folder / "audio.flac")
        wav.unlink()
        entries.append({"index": index, "prompt": prompt, "seed": 5101 + index,
                        "audio": str((folder / "audio.flac").relative_to(args.output)),
                        "reward": reward, "signal": signal,
                        "seconds": rollout["seconds"], "truncated": rollout["truncated"]})
        print(f"evaluation step={step} heldout={index} reward={reward['mean']:.4f}", flush=True)
    write_json(root / "receipt.json", {"optimizer_step": step, "heldout": entries,
                                     "mean_reward": float(np.mean([row["reward"]["mean"] for row in entries]))})


def latest_checkpoint(output):
    ready = [path for path in (output / "checkpoints").glob("step_*")
             if (path / "meta.json").is_file() and (path / "optimizer.pt").is_file()]
    return max(ready, key=lambda path: int(path.name.split("_")[-1])) if ready else None


def save_checkpoint(model, optimizer, step, output):
    parent = output / "checkpoints"
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / f"step_{step:06d}"
    if destination.exists():
        return
    temporary = parent / f".step_{step:06d}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    model.save_pretrained(temporary / "adapter")
    torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
    write_json(temporary / "meta.json", {"optimizer_step": step,
                                          "algorithm": "on-policy two-rollout SongEval GRPO",
                                          "note": "AdamW state resets after the source one-step pilot"})
    temporary.rename(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--start-adapter", type=Path, required=True)
    parser.add_argument("--songeval", type=Path, required=True)
    parser.add_argument("--scorer-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--eval-step", type=int, help="Evaluate a saved adapter without training")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(90730)
    pipe = YuE2Pipeline.from_pretrained(args.model, vae=args.vae, device="cuda",
                                        backend="torch-eager", memory_budget_gib=32,
                                        progress=False)
    base = pipe._load_model().eval()
    evaluate(pipe, 0, args)
    if args.eval_step is not None:
        checkpoint = args.output / "checkpoints" / f"step_{args.eval_step:06d}" / "adapter"
        if not checkpoint.is_dir():
            raise FileNotFoundError(checkpoint)
        PeftModel.from_pretrained(base, checkpoint, is_trainable=False).eval()
        evaluate(pipe, args.eval_step, args)
        return
    resume = latest_checkpoint(args.output)
    adapter_path = resume / "adapter" if resume else args.start_adapter
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=True).eval()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=args.learning_rate)
    start = 1
    if resume:
        optimizer.load_state_dict(torch.load(resume / "optimizer.pt", map_location="cuda", weights_only=False))
        start = int(json.loads((resume / "meta.json").read_text())["optimizer_step"])
    else:
        evaluate(pipe, 1, args)
        save_checkpoint(model, optimizer, 1, args.output)
    if args.max_steps <= start:
        print(f"ready at optimizer step {start}", flush=True)
        return
    log_path = args.output / "steps.jsonl"
    for step in range(start + 1, args.max_steps + 1):
        tick = time.monotonic()
        prompt = TRAIN[(step - 2) % len(TRAIN)]
        scratch = args.output / "scratch"
        if scratch.exists():
            shutil.rmtree(scratch)
        rolls = []
        for index in range(2):
            folder = scratch / f"rollout_{index}"
            wav = folder / "audio.wav"
            rollout = sample(pipe, prompt, 700000 + 2 * step + index, wav, args.max_tokens)
            rollout["reward"] = score_file(args.songeval, args.scorer_python, wav, folder / "reward")
            rolls.append(rollout)
        # Pipeline generation offloads the semantic model to CPU after decoding.
        model.to("cuda")
        rewards = torch.tensor([row["reward"]["mean"] for row in rolls], device="cuda")
        advantages = (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(1e-8)
        with torch.no_grad():
            old = [action_log_probs(model, row, "cuda").detach() for row in rolls]
        optimizer.zero_grad(set_to_none=True)
        for row, previous, advantage in zip(rolls, old, advantages):
            current = action_log_probs(model, row, "cuda")
            ratio = (current - previous).exp()
            loss = -torch.minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage).mean() / 2
            loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), 1.0))
        optimizer.step()
        record = {"optimizer_step": step, "train_prompt_index": (step - 2) % len(TRAIN),
                  "rewards": rewards.tolist(), "gradient_norm": grad_norm,
                  "seconds": time.monotonic() - tick}
        with log_path.open("a") as target:
            target.write(json.dumps(record) + "\n")
        shutil.rmtree(scratch)
        print(f"step={step} rewards={record['rewards']} grad={grad_norm:.4f} seconds={record['seconds']:.1f}", flush=True)
        if step <= 5 or step % args.checkpoint_every == 0 or step in (100, 300, 1000):
            save_checkpoint(model, optimizer, step, args.output)
        if step in (5, 100, 300, 1000):
            evaluate(pipe, step, args)


if __name__ == "__main__":
    main()
