"""One on-policy, two-rollout YuE2 SongEval-GRPO LoRA pilot.

This is an engineering pilot, not a statistically powered comparison. Use a
separate held-out prompt and identical seed for the before/after audio pair.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

from yue2 import YuE2Pipeline
from yue2.protocol import CODEC_OFFSET


TRAIN = {
    "style": "English, bright indie pop, piano, bass, drums, female vocal, 105 BPM",
    "lyrics": "[Verse]\nMorning paints the window gold\nWe let the quiet story unfold",
    "cot": "off", "cfg_scale": 1.0, "id": "train",
}
EVAL = {
    "style": "English, mellow acoustic folk, guitar, light percussion, male vocal, 92 BPM",
    "lyrics": "[Verse]\nUnder open skies we roam\nEvery little road leads home",
    "cot": "off", "cfg_scale": 1.0, "id": "heldout",
}


def generate(pipe, request, seed, output, max_tokens):
    song = pipe(**{**request, "seed": seed}, semantic_sampling={"max_tokens": max_tokens, "min_tokens": 200})
    output.mkdir(parents=True, exist_ok=True)
    song.save_artifacts(output)
    sf.write(output / "audio.wav", song.audio, song.sample_rate, subtype="FLOAT")
    return {
        "prefix": song.semantic.plan.prefix,
        "semantic": song.semantic.tokens,
        "truncated": song.truncated,
        "seconds": len(song.audio) / song.sample_rate,
    }


def score_file(songeval, scorer_python, audio, output):
    subprocess.run([
        str(scorer_python), "eval.py", "-i", str(audio), "-o", str(output)
    ], cwd=songeval, check=True, stdout=subprocess.DEVNULL)
    raw = json.loads((output / "result.json").read_text())[audio.stem]
    return {**raw, "mean": float(np.mean(list(raw.values())))}


def action_log_probs(model, rollout, device):
    ids = rollout["prefix"] + [CODEC_OFFSET + token for token in rollout["semantic"]]
    sequence = torch.tensor(ids, dtype=torch.long, device=device)[None]
    first_action = len(rollout["prefix"]) - 1
    logits = model(input_ids=sequence, use_cache=False).logits[:, first_action:-1]
    targets = sequence[:, first_action + 1:]
    if logits.shape[1] != targets.shape[1] or logits.shape[1] < 2:
        raise AssertionError("Missing semantic actions")
    return -F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                            targets.reshape(-1), reduction="none")


def stats(audio):
    data, _ = sf.read(audio, dtype="float32")
    peak = float(np.max(np.abs(data)))
    return {"peak": peak, "rms": float(np.sqrt(np.mean(data.astype(np.float64) ** 2))),
            "near_full_scale_fraction": float(np.mean(np.abs(data) >= .999)),
            "flat_top_fraction": float(np.mean(np.isclose(np.abs(data), peak, atol=1e-6)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--songeval", type=Path, required=True)
    parser.add_argument("--scorer-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    args = parser.parse_args()
    torch.manual_seed(90729)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    pipe = YuE2Pipeline.from_pretrained(args.model, vae=args.vae, device="cuda",
                                        memory_budget_gib=32, progress=False)
    rollouts = []
    for index, seed in enumerate((4101, 4102)):
        path = out / f"train_{index}"
        roll = generate(pipe, TRAIN, seed, path, args.max_tokens)
        roll["reward"] = score_file(args.songeval, args.scorer_python, path / "audio.wav", path / "reward")
        rollouts.append(roll)
        print(f"train rollout {index}: reward={roll['reward']['mean']:.4f}", flush=True)
    before = out / "heldout_before"
    before_roll = generate(pipe, EVAL, 5101, before, args.max_tokens)
    before_reward = score_file(args.songeval, args.scorer_python, before / "audio.wav", before / "reward")
    print(f"heldout before: reward={before_reward['mean']:.4f}", flush=True)

    model = get_peft_model(pipe._load_model(), LoraConfig(
        task_type="CAUSAL_LM", r=4, lora_alpha=8, lora_dropout=0,
        target_modules=("q_proj", "v_proj"), bias="none"))
    model.eval()
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad),
                                  lr=args.learning_rate)
    rewards = torch.tensor([roll["reward"]["mean"] for roll in rollouts], dtype=torch.float32, device="cuda")
    advantages = (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(1e-8)
    with torch.no_grad():
        old = [action_log_probs(model, roll, "cuda").detach() for roll in rollouts]
    optimizer.zero_grad(set_to_none=True)
    for roll, previous, advantage in zip(rollouts, old, advantages):
        current = action_log_probs(model, roll, "cuda")
        ratio = (current - previous).exp()
        clipped = ratio.clamp(.8, 1.2)
        loss = -torch.minimum(ratio * advantage, clipped * advantage).mean() / len(rollouts)
        loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(
        (parameter for parameter in model.parameters() if parameter.requires_grad), 1.0))
    optimizer.step()
    with torch.no_grad():
        mean_action_delta = float(torch.stack([
            (action_log_probs(model, roll, "cuda") - previous).mean()
            for roll, previous in zip(rollouts, old)
        ]).abs().mean())
    model.save_pretrained(out / "adapter")
    pipe._model = model.merge_and_unload().eval()
    del model
    torch.cuda.empty_cache()
    after = out / "heldout_after"
    after_roll = generate(pipe, EVAL, 5101, after, args.max_tokens)
    after_reward = score_file(args.songeval, args.scorer_python, after / "audio.wav", after / "reward")
    with torch.no_grad():
        print(f"heldout after: reward={after_reward['mean']:.4f}", flush=True)
    receipt = {
        "status": "on_policy_two_rollout_one_step_grpo_pilot",
        "train_rollout_rewards": [roll["reward"] for roll in rollouts],
        "train_rollout_lengths": [len(roll["semantic"]) for roll in rollouts],
        "advantages": advantages.tolist(),
        "gradient_norm": grad_norm,
        "mean_action_logprob_delta": mean_action_delta,
        "heldout_prompt": "distinct original short English folk verse",
        "heldout_seed": 5101,
        "before": {"reward": before_reward, "signal": stats(before / "audio.wav"),
                   "seconds": before_roll["seconds"], "truncated": before_roll["truncated"]},
        "after": {"reward": after_reward, "signal": stats(after / "audio.wav"),
                  "seconds": after_roll["seconds"], "truncated": after_roll["truncated"]},
        "caveat": "Single GRPO step, 2 rollouts and 1 held-out seed; audio is capped and may be truncated. No generalization claim.",
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
