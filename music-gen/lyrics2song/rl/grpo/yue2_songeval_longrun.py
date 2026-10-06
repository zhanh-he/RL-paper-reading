"""Resume a small-prompt on-policy YuE2 SongEval-GRPO run from the one-step LoRA.

This experiment uses optimizer updates as its step count. It saves fixed-seed
held-out audio and adapters at milestones; it does not assign quality labels to
steps before listening and measuring them.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from peft import LoraConfig, PeftModel, get_peft_model

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
    return {"prefix": [int(token) for token in song.semantic.plan.prefix],
            "semantic": [int(token) for token in song.semantic.tokens],
            "seconds": len(song.audio) / song.sample_rate, "truncated": song.truncated}


def score_many(pipe, args, items):
    if args.reward_backend == "songeval":
        if not args.dataset_manifest:
            return [score_file(args.songeval, args.scorer_python, wav, folder)
                    for wav, folder in items]
        batch = items[0][1].parent.parent / "songeval_batch"
        inputs = batch / "inputs"
        inputs.mkdir(parents=True, exist_ok=True)
        paths = []
        for index, (wav, _) in enumerate(items):
            link = inputs / f"audio_{index:04d}.wav"
            if not link.exists():
                link.symlink_to(wav.resolve())
            paths.append(link)
        (batch / "inputs.txt").write_text("".join(f"{path.absolute()}\n" for path in paths))
        if pipe._model is not None:
            pipe._model.to("cpu")
        if pipe._vae is not None:
            pipe._vae.to("cpu")
        torch.cuda.empty_cache()
        subprocess.run([str(args.scorer_python), "eval.py", "-i", str(batch / "inputs.txt"),
                        "-o", str(batch / "scores")], cwd=args.songeval, check=True)
        scored = json.loads((batch / "scores" / "result.json").read_text())
        return [{**scored[path.stem], "mean": float(np.mean(list(scored[path.stem].values())))}
                for path in paths]
    canonical_items = []
    for wav, folder in items:
        audio, sr = sf.read(wav, dtype="float32")
        canonical = wav.with_suffix(".flac")
        sf.write(canonical, audio, sr, subtype="PCM_24")
        canonical_items.append((canonical, folder))
    if pipe._model is not None:
        pipe._model.to("cpu")
    if pipe._vae is not None:
        pipe._vae.to("cpu")
    torch.cuda.empty_cache()
    command = [str(args.musecritic_python), str(Path(__file__).with_name("score_musecritic_batch.py")),
               "--repo", str(args.musecritic_repo), "--model", str(args.musecritic_model),
               "--max-new-tokens", str(args.musecritic_max_new_tokens)]
    for canonical, folder in canonical_items:
        command.extend(["--pair", str(canonical), str(folder)])
    subprocess.run(command, check=True)
    rewards = []
    for canonical, folder in canonical_items:
        scores = json.loads((folder / "result.json").read_text())[canonical.stem]
        rewards.append({**scores, "mean": float(np.mean(list(scores.values())))})
    return rewards


def evaluate(pipe, step, args, heldout):
    root = args.output / f"step_{step:06d}"
    if (root / "receipt.json").exists():
        return
    entries = []
    pending = []
    for index, prompt in enumerate(heldout):
        folder = root / f"heldout_{index}"
        wav = folder / "audio.wav"
        rollout = sample(pipe, prompt, args.eval_seed_base + index, wav, args.max_tokens)
        pending.append((wav, folder / "reward"))
        entries.append({"index": index, "prompt": prompt, "seed": args.eval_seed_base + index,
                        "audio": str((folder / "audio.flac").relative_to(args.output)),
                        "seconds": rollout["seconds"], "truncated": rollout["truncated"]})
    rewards = score_many(pipe, args, pending)
    for entry, (wav, _), reward in zip(entries, pending, rewards):
        folder = wav.parent
        canonical = folder / "audio.flac"
        scored = canonical if args.reward_backend == "musecritic" else wav
        entry["scored_audio"] = str(scored.relative_to(args.output))
        entry["scored_audio_sha256"] = hashlib.sha256(scored.read_bytes()).hexdigest()
        if args.reward_backend == "songeval":
            audio, sr = sf.read(wav, dtype="float32")
            sf.write(canonical, audio, sr, subtype="PCM_24")
        signal = stats(canonical)
        if not args.dataset_manifest:
            wav.unlink()
        entry.update({"reward": reward, "signal": signal,
                      "audio_sha256": hashlib.sha256(canonical.read_bytes()).hexdigest()})
        print(f"evaluation step={step} heldout={entry['index']} reward={reward['mean']:.4f}", flush=True)
    write_json(root / "receipt.json", {"optimizer_step": step, "heldout": entries,
                                     "mean_reward": float(np.mean([row["reward"]["mean"] for row in entries]))})


def latest_checkpoint(output):
    ready = [path for path in (output / "checkpoints").glob("step_*")
             if (path / "meta.json").is_file() and (path / "optimizer.pt").is_file()]
    return max(ready, key=lambda path: int(path.name.split("_")[-1])) if ready else None


def save_checkpoint(model, optimizer, step, output, reward_backend="songeval"):
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
                                          "algorithm": f"on-policy two-rollout {reward_backend} GRPO",
                                          "note": "AdamW state resets after the source one-step pilot"})
    temporary.rename(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--start-adapter", type=Path)
    parser.add_argument("--dataset-manifest", type=Path, help="Frozen CMI-Pref prompt manifest; starts from a fresh LoRA")
    parser.add_argument("--expected-dataset-sha256", help="Required with --dataset-manifest")
    parser.add_argument("--eval-seed-base", type=int, default=5101)
    parser.add_argument("--smoke", action="store_true", help="Separate two-condition, two-step wiring test; never a formal result")
    parser.add_argument("--reward-backend", choices=("songeval", "musecritic"), default="songeval")
    parser.add_argument("--songeval", type=Path)
    parser.add_argument("--scorer-python", type=Path)
    parser.add_argument("--musecritic-python", type=Path)
    parser.add_argument("--musecritic-repo", type=Path)
    parser.add_argument("--musecritic-model", type=Path)
    parser.add_argument("--musecritic-max-new-tokens", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--kl-beta", type=float, default=0.0,
                        help="Sampled reverse-KL penalty against the frozen base; required for formal runs")
    parser.add_argument("--compute-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--eval-step", type=int, help="Evaluate a saved adapter without training")
    args = parser.parse_args()
    if args.reward_backend == "songeval" and (args.songeval is None or args.scorer_python is None):
        parser.error("SongEval requires --songeval and --scorer-python")
    if args.reward_backend == "musecritic" and any(value is None for value in
            (args.musecritic_python, args.musecritic_repo, args.musecritic_model)):
        parser.error("MuseCritic requires --musecritic-python, --musecritic-repo and --musecritic-model")
    if args.dataset_manifest:
        if args.start_adapter or not args.expected_dataset_sha256:
            parser.error("Formal data requires its SHA256 and a fresh LoRA, not --start-adapter")
        if args.kl_beta <= 0:
            parser.error("Formal data requires --kl-beta > 0 for logged KL regularization")
        actual_sha = hashlib.sha256(args.dataset_manifest.read_bytes()).hexdigest()
        if actual_sha != args.expected_dataset_sha256:
            parser.error(f"Dataset SHA256 changed: {actual_sha}")
        dataset = json.loads(args.dataset_manifest.read_text())
        protocol = dataset.get("protocol")
        if protocol not in ("cmi-pref-lyrics-no-ref-decontaminated-v1",
                            "cmi-pref-triple-source-text-lyrics-baseline-v1"):
            parser.error("Unrecognized formal dataset protocol")
        if protocol == "cmi-pref-triple-source-text-lyrics-baseline-v1":
            if dataset.get("source_manifest_sha256") != "b611333ef0abdeaf7a0473ea0beaa470673c3414f05b816a97904d2ade25b2e0" \
                    or dataset.get("input_modalities") != ["text", "lyrics"] \
                    or dataset.get("reference_audio_used") is not False:
                parser.error("Three-part source projection is not the audited text+lyrics baseline")
        train = [{"style": row["style"], "lyrics": row["lyrics"], "cot": "off", "cfg_scale": 1.0}
                 for row in dataset["train"]]
        heldout = [{"style": row["style"], "lyrics": row["lyrics"], "cot": "off", "cfg_scale": 1.0}
                   for row in dataset["valid"]]
        expected_counts = (240, 60) if protocol == "cmi-pref-triple-source-text-lyrics-baseline-v1" else (234, 59)
        if (len(train), len(heldout)) != expected_counts:
            parser.error("Formal split size changed")
        if args.smoke:
            train, heldout = train[:2], heldout[:2]
            if args.max_steps > 2:
                parser.error("Smoke run must have at most two steps")
    else:
        if not args.start_adapter or args.expected_dataset_sha256 or args.smoke:
            parser.error("Legacy pilot requires --start-adapter; smoke requires --dataset-manifest")
        train, heldout = TRAIN, HELDOUT
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "training_data_origin": ("pinned CMI-Pref triple-source text+lyrics projection" if args.dataset_manifest
                                  and protocol == "cmi-pref-triple-source-text-lyrics-baseline-v1"
                                  else "pinned CMI-Pref compatible train subset" if args.dataset_manifest
                                  else "eight original hand-written English two-line lyrics prompts"),
        "heldout_data_origin": "pinned CMI-Pref train-derived validation subset" if args.dataset_manifest else "three separate original hand-written English two-line lyrics prompts",
        "dataset_sha256": args.expected_dataset_sha256,
        "dataset_protocol": protocol if args.dataset_manifest else None,
        "input_modalities": ["text", "lyrics"] if args.dataset_manifest else None,
        "reference_audio_used": False if args.dataset_manifest else None,
        "source_manifest_sha256": dataset.get("source_manifest_sha256") if args.dataset_manifest else None,
        "kl_beta": args.kl_beta if args.dataset_manifest else None,
        "compute_dtype": args.compute_dtype if args.dataset_manifest else None,
        "smoke_only": args.smoke,
        "train_prompt_count": len(train), "validation_prompt_count": len(heldout),
        "train_prompts": train if not args.dataset_manifest else None,
        "heldout_prompts": heldout if not args.dataset_manifest else None,
        "eval_seed_base": args.eval_seed_base,
        "reward_backend": args.reward_backend, "learning_rate": args.learning_rate,
        "reward_audio_format": "PCM_24 FLAC" if args.reward_backend == "musecritic" else "FLOAT WAV",
        "max_tokens": args.max_tokens, "start_adapter": str(args.start_adapter) if args.start_adapter else None,
        "musecritic_model": str(args.musecritic_model) if args.musecritic_model else None,
        "musecritic_max_new_tokens": args.musecritic_max_new_tokens if args.reward_backend == "musecritic" else None,
    }
    if not args.dataset_manifest:
        for key in ("dataset_sha256", "dataset_protocol", "input_modalities", "reference_audio_used",
                    "source_manifest_sha256", "kl_beta", "compute_dtype", "smoke_only", "train_prompt_count",
                    "validation_prompt_count", "eval_seed_base"):
            manifest.pop(key)
    manifest_path = args.output / "experiment.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError(f"Run configuration changed: {manifest_path}")
    if not manifest_path.exists():
        write_json(manifest_path, manifest)
    torch.manual_seed(90730)
    pipe = YuE2Pipeline.from_pretrained(args.model, vae=args.vae, device="cuda",
                                        backend="torch-eager", memory_budget_gib=32,
                                        progress=False)
    base = pipe._load_model().eval()
    actual_dtype = next(base.parameters()).dtype
    expected_dtype = torch.bfloat16 if args.compute_dtype == "bf16" else torch.float16
    if args.dataset_manifest and actual_dtype != expected_dtype:
        raise ValueError(f"Model dtype {actual_dtype} != declared {expected_dtype}")
    evaluate(pipe, 0, args, heldout)
    if args.eval_step is not None:
        checkpoint = args.output / "checkpoints" / f"step_{args.eval_step:06d}" / "adapter"
        if not checkpoint.is_dir():
            raise FileNotFoundError(checkpoint)
        PeftModel.from_pretrained(base, checkpoint, is_trainable=False).eval()
        evaluate(pipe, args.eval_step, args, heldout)
        return
    resume = latest_checkpoint(args.output)
    if resume or args.start_adapter:
        adapter_path = resume / "adapter" if resume else args.start_adapter
        model = PeftModel.from_pretrained(base, adapter_path, is_trainable=True).eval()
    else:
        model = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", r=4,
            lora_alpha=8, lora_dropout=0, target_modules=("q_proj", "v_proj"), bias="none")).eval()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=args.learning_rate)
    start = 0 if args.dataset_manifest else 1
    if resume:
        optimizer.load_state_dict(torch.load(resume / "optimizer.pt", map_location="cuda", weights_only=False))
        start = int(json.loads((resume / "meta.json").read_text())["optimizer_step"])
    elif args.dataset_manifest:
        save_checkpoint(model, optimizer, 0, args.output, args.reward_backend)
    else:
        evaluate(pipe, 1, args, heldout)
        save_checkpoint(model, optimizer, 1, args.output, args.reward_backend)
    if args.max_steps <= start:
        print(f"ready at optimizer step {start}", flush=True)
        return
    log_path = args.output / "steps.jsonl"
    for step in range(start + 1, args.max_steps + 1):
        tick = time.monotonic()
        train_index = (step - 1) % len(train) if args.dataset_manifest else (step - 2) % len(train)
        prompt = train[train_index]
        scratch = args.output / "scratch"
        if scratch.exists():
            shutil.rmtree(scratch)
        rolls = []
        for index in range(2):
            folder = scratch / f"rollout_{index}"
            wav = folder / "audio.wav"
            rollout = sample(pipe, prompt, 700000 + 2 * step + index, wav, args.max_tokens)
            rolls.append(rollout)
        rewards_scored = score_many(pipe, args, [(scratch / f"rollout_{index}" / "audio.wav",
                                                  scratch / f"rollout_{index}" / "reward")
                                                 for index in range(2)])
        for rollout, reward in zip(rolls, rewards_scored):
            rollout["reward"] = reward
        # Pipeline generation offloads the semantic model to CPU after decoding.
        model.to("cuda")
        rewards = torch.tensor([row["reward"]["mean"] for row in rolls], device="cuda")
        if not torch.isfinite(rewards).all():
            raise FloatingPointError(f"Nonfinite reward at step {step}")
        advantages = (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(1e-8)
        with torch.no_grad():
            old = [action_log_probs(model, row, "cuda").detach() for row in rolls]
            if args.dataset_manifest:
                with model.disable_adapter():
                    reference = [action_log_probs(model, row, "cuda").detach() for row in rolls]
            else:
                reference = None
        optimizer.zero_grad(set_to_none=True)
        sampled_kl = []
        for index, (row, previous, advantage) in enumerate(zip(rolls, old, advantages)):
            current = action_log_probs(model, row, "cuda")
            ratio = (current - previous).exp()
            loss = -torch.minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage).mean() / 2
            if reference is not None:
                log_ref_ratio = (reference[index] - current).clamp(-20, 20)
                kl = (log_ref_ratio.exp() - log_ref_ratio - 1).mean()
                sampled_kl.append(float(kl.detach()))
                loss = loss + args.kl_beta * kl / 2
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite policy loss at step {step}")
            loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), 1.0))
        if not np.isfinite(grad_norm):
            raise FloatingPointError(f"Nonfinite gradient norm at step {step}")
        optimizer.step()
        if any(not torch.isfinite(parameter).all() for parameter in model.parameters()
               if parameter.requires_grad):
            raise FloatingPointError(f"Nonfinite adapter parameter at step {step}")
        record = {"optimizer_step": step, "train_prompt_index": train_index,
                  "rewards": rewards.tolist(), "reward_components": rewards_scored,
                  "sampled_kl": float(np.mean(sampled_kl)) if sampled_kl else None,
                  "kl_beta": args.kl_beta if args.dataset_manifest else None,
                  "gradient_norm": grad_norm,
                  "seconds": time.monotonic() - tick}
        if args.dataset_manifest:
            archive = args.output / "train_rollouts" / f"step_{step:06d}"
            archive.parent.mkdir(parents=True, exist_ok=True)
            if archive.exists():
                raise FileExistsError(f"Refusing to overwrite scored rollout audio: {archive}")
            record["scored_audio"] = []
            for index, rollout in enumerate(rolls):
                scored = scratch / f"rollout_{index}" / ("audio.flac" if args.reward_backend == "musecritic" else "audio.wav")
                if not scored.is_file():
                    raise FileNotFoundError(scored)
                record["scored_audio"].append({
                    "path": str((archive / f"rollout_{index}" / scored.name).relative_to(args.output)),
                    "sha256": hashlib.sha256(scored.read_bytes()).hexdigest(),
                })
                write_json(scratch / f"rollout_{index}" / "rollout.json", rollout)
            scratch.rename(archive)
        with log_path.open("a") as target:
            target.write(json.dumps(record) + "\n")
        if not args.dataset_manifest:
            shutil.rmtree(scratch)
        print(f"step={step} rewards={record['rewards']} grad={grad_norm:.4f} seconds={record['seconds']:.1f}", flush=True)
        if step <= 5 or step % args.checkpoint_every == 0 or step in (100, 300, 1000):
            save_checkpoint(model, optimizer, step, args.output, args.reward_backend)
        if args.dataset_manifest and step in (1, 25, 50, 100):
            evaluate(pipe, step, args, heldout)
        elif not args.dataset_manifest and step in (5, 100, 300, 1000):
            evaluate(pipe, step, args, heldout)


if __name__ == "__main__":
    main()
