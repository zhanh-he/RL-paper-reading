"""Replay a fixed YuE2-matched prompt for a Muse LoRA checkpoint."""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel

from muse_songeval_pilot import decode, load, rollout, score, signal


PROMPT = (
    "Please generate a song in the following style: English, mellow acoustic folk, "
    "guitar, light percussion, male vocal, 92 BPM.\n"
    "[Verse][lyrics:\nUnder open skies we roam\nEvery little road leads home]"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--mucodec", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--codec-python", type=Path, required=True)
    parser.add_argument("--songeval", type=Path, required=True)
    parser.add_argument("--scorer-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=5101)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--optimizer-step", type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    pairs = {}
    for name in ("before", "after"):
        model, tokenizer = load(args.model)
        if name == "after":
            model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload().eval()
        path = args.output / f"heldout_{name}.json"
        row = rollout(model, tokenizer, PROMPT, args.seed, args.max_tokens, path, force_length=True)
        if not row["audio_tokens"]:
            raise RuntimeError(f"No Muse audio tokens for {name}")
        row["audio_token_start"] = tokenizer.convert_tokens_to_ids("<AUDIO_0>")
        path.write_text(json.dumps(row))
        del model
        torch.cuda.empty_cache()
        decode(args, [path])
        wav = path.with_suffix(".wav")
        pairs[name] = {
            "reward": score(args, wav, args.output / f"heldout_{name}_reward"),
            "signal": signal(wav),
            "generated_tokens": len(row["generated_ids"]),
            "audio_tokens": len(row["audio_tokens"]),
        }
        print(f"{name}: SongEval mean={pairs[name]['reward']['mean']:.4f}", flush=True)

    receipt = {
        "status": "muse_songeval_yue2_matched_heldout_pair",
        "model": "Muse-0.6b + MuCodec",
        "reward_model": "SongEval",
        "optimizer_step": args.optimizer_step,
        "heldout_prompt": PROMPT,
        "heldout_style": "English, mellow acoustic folk, guitar, light percussion, male vocal, 92 BPM",
        "heldout_lyrics": "[Verse]\nUnder open skies we roam\nEvery little road leads home",
        "heldout_seed": args.seed,
        "max_new_tokens": args.max_tokens,
        "force_length": True,
        "before": pairs["before"],
        "after": pairs["after"],
        "caveat": "One short held-out prompt/seed; score changes are not a human-quality claim.",
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
