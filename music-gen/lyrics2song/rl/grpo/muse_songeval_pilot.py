"""Staged on-policy Muse/MuCodec/SongEval GRPO pilot with held-out audio pair."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


TRAIN_PROMPT = (
    "Please generate a song in the following style: English indie pop, bright piano, "
    "round bass, light drums, female vocal, 105 BPM.\n"
    "[Verse][desc:Warm and melodic with a clear vocal above a restrained band.]"
    "[lyrics:\nMorning paints the window gold\nWe let the quiet story unfold]"
)
HELDOUT_PROMPT = (
    "Please generate a song in the following style: English acoustic folk, "
    "fingerpicked guitar, soft percussion, male vocal, 92 BPM.\n"
    "[Verse][desc:Intimate melody and natural acoustic accompaniment.]"
    "[lyrics:\nUnder open skies we roam\nEvery little road leads home]"
)


def load(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, trust_remote_code=True,
        local_files_only=True, attn_implementation="sdpa",
    ).to("cuda").eval()
    return model, tokenizer


def rollout(model, tokenizer, prompt, seed, max_tokens, path, force_length=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                        tokenize=True, add_generation_prompt=True)
    prompt_ids = torch.tensor([ids], device="cuda")
    with torch.inference_mode():
        generated = model.generate(
            input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids),
            do_sample=True, temperature=.9, top_p=.9,
            repetition_penalty=1.3, max_new_tokens=max_tokens,
            min_new_tokens=max_tokens if force_length else 0,
            pad_token_id=tokenizer.eos_token_id,
        )[0, len(ids):].tolist()
    start = tokenizer.convert_tokens_to_ids("<AUDIO_0>")
    end = tokenizer.convert_tokens_to_ids("<AUDIO_16383>")
    if end - start != 16383:
        raise ValueError("Muse audio token IDs are not contiguous")
    audio = [value - start for value in generated if start <= value <= end]
    row = {"prompt_ids": ids, "generated_ids": generated, "audio_tokens": audio,
           "seed": seed, "max_new_tokens": max_tokens, "force_length": force_length}
    path.write_text(json.dumps(row))
    print(f"rollout {path.name}: {len(generated)} tokens, {len(audio)} audio tokens", flush=True)
    return row


def decode(args, files):
    subprocess.run([
        str(args.codec_python), str(args.decoder), "--mucodec", str(args.mucodec),
        "--steps", "20", "--token-file", *map(str, files),
    ], cwd=args.mucodec, check=True)


def score(args, wav, destination):
    subprocess.run([str(args.scorer_python), "eval.py", "-i", str(wav), "-o", str(destination)],
                   cwd=args.songeval, check=True, stdout=subprocess.DEVNULL)
    values = json.loads((destination / "result.json").read_text())[wav.stem]
    return {**values, "mean": float(np.mean(list(values.values())))}


def log_probs(model, row):
    ids = torch.tensor([row["prompt_ids"] + row["generated_ids"]], device="cuda")
    first = len(row["prompt_ids"]) - 1
    logits = model(input_ids=ids, use_cache=False).logits[:, first:-1].float()
    targets = ids[:, first + 1:]
    start = row["audio_token_start"]
    mask = (targets >= start) & (targets <= start + 16383)
    if mask.sum() < 32:
        raise ValueError("Not enough Muse audio actions for GRPO")
    return -F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                            targets.reshape(-1), reduction="none")[mask.reshape(-1)]


def signal(path):
    audio, _ = sf.read(path, dtype="float32")
    peak = float(np.max(np.abs(audio)))
    return {"seconds": len(audio) / 48000, "peak": peak,
            "rms": float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))),
            "near_full_scale_fraction": float(np.mean(np.abs(audio) >= .999)),
            "flat_top_fraction": float(np.mean(np.isclose(np.abs(audio), peak, atol=1e-6)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("baseline", "train", "after"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mucodec", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--codec-python", type=Path, required=True)
    parser.add_argument("--songeval", type=Path, required=True)
    parser.add_argument("--scorer-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--force-length", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load(args.model)
    audio_start = tokenizer.convert_tokens_to_ids("<AUDIO_0>")

    if args.phase == "baseline":
        specs = (("train_0", TRAIN_PROMPT, 4101), ("train_1", TRAIN_PROMPT, 4102),
                 ("heldout_before", HELDOUT_PROMPT, 5101),
                 ("heldout_replay", HELDOUT_PROMPT, 5101))
        files = []
        for name, prompt, seed in specs:
            path = args.output / f"{name}.json"
            row = rollout(model, tokenizer, prompt, seed, args.max_tokens, path, args.force_length)
            row["audio_token_start"] = audio_start
            path.write_text(json.dumps(row))
            files.append(path)
        assert json.loads(files[2].read_text())["generated_ids"] == json.loads(files[3].read_text())["generated_ids"]
        del model
        torch.cuda.empty_cache()
        decode(args, files)
        baseline = {}
        for path in files:
            wav = path.with_suffix(".wav")
            baseline[path.stem] = {"reward": score(args, wav, args.output / f"{path.stem}_reward"),
                                   "signal": signal(wav)}
        a, _ = sf.read(args.output / "heldout_before.wav", dtype="float32")
        b, _ = sf.read(args.output / "heldout_replay.wav", dtype="float32")
        baseline["untrained_replay_audio_identical"] = bool(np.array_equal(a, b))
        (args.output / "baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
        print(json.dumps(baseline, indent=2), flush=True)
        return

    if args.phase == "train":
        baseline = json.loads((args.output / "baseline.json").read_text())
        group = [json.loads((args.output / f"train_{index}.json").read_text()) for index in (0, 1)]
        reward = torch.tensor([baseline[f"train_{index}"]["reward"]["mean"] for index in (0, 1)],
                              device="cuda")
        advantage = (reward - reward.mean()) / reward.std(unbiased=False).clamp_min(1e-8)
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=8, lora_alpha=16,
                                                target_modules=("q_proj", "v_proj"), lora_dropout=0,
                                                bias="none")).eval()
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                      lr=args.learning_rate)
        with torch.no_grad():
            old = [log_probs(model, row).detach() for row in group]
        optimizer.zero_grad(set_to_none=True)
        for row, previous, weight in zip(group, old, advantage):
            current = log_probs(model, row)
            ratio = (current - previous).exp()
            loss = -torch.minimum(ratio * weight, ratio.clamp(.8, 1.2) * weight).mean() / 2
            loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1))
        optimizer.step()
        with torch.no_grad():
            delta = float(torch.stack([(log_probs(model, row) - previous).mean().abs()
                                       for row, previous in zip(group, old)]).mean())
        model.save_pretrained(args.output / "adapter")
        (args.output / "training.json").write_text(json.dumps({
            "algorithm": "on-policy GRPO, 2 rollouts, 1 LoRA optimizer step",
            "group_rewards": reward.tolist(), "advantages": advantage.tolist(),
            "gradient_norm": grad_norm, "mean_action_logprob_delta": delta,
        }, indent=2) + "\n")
        print((args.output / "training.json").read_text(), flush=True)
        return

    model = PeftModel.from_pretrained(model, args.output / "adapter").merge_and_unload().eval()
    path = args.output / "heldout_after.json"
    row = rollout(model, tokenizer, HELDOUT_PROMPT, 5101, args.max_tokens, path, args.force_length)
    row["audio_token_start"] = audio_start
    path.write_text(json.dumps(row))
    del model
    torch.cuda.empty_cache()
    decode(args, [path])
    before = json.loads((args.output / "baseline.json").read_text())
    after = {"reward": score(args, path.with_suffix(".wav"), args.output / "heldout_after_reward"),
             "signal": signal(path.with_suffix(".wav"))}
    before_ids = json.loads((args.output / "heldout_before.json").read_text())["generated_ids"]
    receipt = {
        "status": "on_policy_two_rollout_one_step_grpo_pilot",
        "model": "Muse-0.6b + MuCodec released checkpoints",
        "codec_steps": 20, "max_new_tokens": args.max_tokens, "force_length": args.force_length,
        "training": json.loads((args.output / "training.json").read_text()),
        "heldout_seed": 5101,
        "untrained_replay_audio_identical": before["untrained_replay_audio_identical"],
        "heldout_generated_token_difference": sum(x != y for x, y in zip(before_ids, row["generated_ids"])),
        "before": before["heldout_before"], "after": after,
        "caveat": "One prompt/group, two rollouts, one optimizer step and one held-out prompt. Short capped fragment, not a statistical song-quality result.",
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
