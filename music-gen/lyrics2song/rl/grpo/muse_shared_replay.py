"""Generate shared-prompt Muse token replays for cross-reward A/B listening."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from peft import PeftModel

from muse_songeval_pilot import load, rollout


PROMPT = (
    "Please generate a song in the following style: English, mellow acoustic folk, "
    "guitar, light percussion, male vocal, 92 BPM.\n"
    "[Verse][lyrics:\nUnder open skies we roam\nEvery little road leads home]"
)
STAGES = {
    "baseline": None,
    "songeval-1": "songeval-1",
    "musecritic-1": "musecritic-1",
    "musecritic-25": "checkpoint-25",
    "musecritic-50": "checkpoint-50",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=5101)
    parser.add_argument("--max-tokens", type=int, default=500)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    stages = {}
    baseline_ids = None
    for name, adapter_name in STAGES.items():
        model, tokenizer = load(args.model)
        adapter = args.adapters / adapter_name if adapter_name else None
        if adapter:
            model = PeftModel.from_pretrained(model, adapter).merge_and_unload().eval()
        token_file = args.output / f"{name}_tokens.json"
        row = rollout(model, tokenizer, PROMPT, args.seed, args.max_tokens,
                      token_file, force_length=True)
        if not row["audio_tokens"]:
            raise RuntimeError(f"{name} produced no audio tokens")
        if name == "baseline":
            baseline_ids = row["generated_ids"]
        stages[name] = {
            "adapter": str(adapter) if adapter else None,
            "adapter_sha256": sha256(adapter / "adapter_model.safetensors") if adapter else None,
            "token_file": token_file.name,
            "generated_tokens": len(row["generated_ids"]),
            "audio_tokens": len(row["audio_tokens"]),
            "changed_tokens_vs_baseline": sum(a != b for a, b in zip(baseline_ids, row["generated_ids"])),
        }
        del model
        torch.cuda.empty_cache()
    receipt = {
        "status": "shared_muse_prompt_token_replay",
        "model": str(args.model),
        "prompt": PROMPT,
        "seed": args.seed,
        "max_new_tokens": args.max_tokens,
        "force_length": True,
        "sampling": {"temperature": 0.9, "top_p": 0.9, "repetition_penalty": 1.3},
        "stages": stages,
        "note": "Tokens only; decoded audio and reward scores require separate verification.",
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
