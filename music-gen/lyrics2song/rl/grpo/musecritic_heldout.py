"""Replay a fixed Muse prompt before/after a one-step MuseCritic-GRPO adapter."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from peft import PeftModel

from muse_songeval_pilot import HELDOUT_PROMPT, load, rollout, signal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=5101)
    parser.add_argument("--max-tokens", type=int, default=500)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.plugin.parent.resolve()))
    from plugin import MusecriticRM

    critic = MusecriticRM()
    pairs = {}
    for name in ("before", "after"):
        model, tokenizer = load(args.model)
        if name == "after":
            model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload().eval()
        token_file = args.output / f"heldout_{name}_tokens.json"
        row = rollout(model, tokenizer, HELDOUT_PROMPT, args.seed, args.max_tokens,
                      token_file, force_length=True)
        if not row["audio_tokens"]:
            raise RuntimeError(f"{name} has no Muse audio tokens")
        del model
        torch.cuda.empty_cache()
        completion = "".join(f"<AUDIO_{token}>" for token in row["audio_tokens"])
        decoded = critic._decode_completion_to_wav(completion, f"heldout_{name}", 0)
        audio = args.output / f"heldout_{name}.wav"
        shutil.copyfile(decoded, audio)
        reward, detail = critic._score_audio(str(audio))
        pairs[name] = {
            "reward": {**detail["reward_scores"], "mean": reward},
            "signal": signal(audio),
            "generated_tokens": len(row["generated_ids"]),
            "audio_tokens": len(row["audio_tokens"]),
            "token_ids": row["generated_ids"],
        }
        print(f"{name}: reward={reward:.4f}, audio_tokens={len(row['audio_tokens'])}", flush=True)
    difference = sum(a != b for a, b in zip(pairs["before"]["token_ids"], pairs["after"]["token_ids"]))
    receipt = {
        "status": "one_step_musecritic_grpo_heldout_pair",
        "model": "Muse-0.6b + MuCodec",
        "reward_model": "MuseCritic",
        "heldout_prompt": HELDOUT_PROMPT,
        "heldout_seed": args.seed,
        "generated_token_difference": difference,
        "before": {key: value for key, value in pairs["before"].items() if key != "token_ids"},
        "after": {key: value for key, value in pairs["after"].items() if key != "token_ids"},
        "caveat": "One short held-out prompt/seed after one small online GRPO update; reward and signal are not a human-quality or generalization claim.",
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
