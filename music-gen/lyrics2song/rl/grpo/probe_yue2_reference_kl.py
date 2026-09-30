"""Measure offline codec-constrained conditional KL from the fixed step-1 policy."""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from yue2 import YuE2Pipeline
from yue2.protocol import CODEC_OFFSET, CODEC_SIZE, MUSIC_END

from yue2_songeval_longrun import HELDOUT, write_json


@torch.inference_mode()
def constrained_log_probs(model, rollout):
    ids = rollout["prefix"] + [CODEC_OFFSET + token for token in rollout["semantic"]]
    sequence = torch.tensor(ids, dtype=torch.long, device="cuda")[None]
    first_action = len(rollout["prefix"]) - 1
    logits = model(input_ids=sequence, use_cache=False).logits[:, first_action:-1].float()
    if logits.shape[1] != len(rollout["semantic"]):
        raise AssertionError("Semantic action length mismatch")
    codec = logits[..., CODEC_OFFSET:CODEC_OFFSET + CODEC_SIZE]
    end = logits[..., MUSIC_END:MUSIC_END + 1].clone()
    end[:, :min(200, end.shape[1])] = float("-inf")
    allowed = torch.cat((codec, end), dim=-1)
    allowed_mass = (torch.logsumexp(allowed, dim=-1) -
                    torch.logsumexp(logits, dim=-1)).exp().mean().item()
    return torch.log_softmax(allowed, dim=-1).squeeze(0).cpu(), allowed_mass


def conditional_kl(reference, comparison):
    valid = torch.isfinite(reference)
    delta = torch.where(valid, reference - comparison, 0.0)
    value = (reference.exp() * delta).sum(-1).mean().item()
    if not torch.isfinite(torch.tensor(value)):
        raise FloatingPointError("Nonfinite conditional KL")
    return value


def reference_rollouts(pipe, output, max_tokens):
    path = output / "reference_tokens.json"
    if path.exists():
        records = json.loads(path.read_text())
        if [row["seed"] for row in records] != [5101, 5102, 5103]:
            raise ValueError("Reference probe seeds changed")
        return records
    records = []
    for index, prompt in enumerate(HELDOUT):
        seed = 5101 + index
        with torch.inference_mode():
            plan = pipe.plan(**{**prompt, "seed": seed})
            semantic = pipe.generate_semantic(
                plan, sampling={"max_tokens": max_tokens, "min_tokens": 200})
        records.append({"seed": seed, "prompt": prompt, "prefix": plan.prefix,
                        "semantic": semantic.tokens})
        print(f"reference probe {index}: {len(semantic.tokens)} actions", flush=True)
    write_json(path, records)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--reference-adapter", type=Path, required=True)
    parser.add_argument("--arm", nargs=2, action="append", required=True,
                        metavar=("NAME", "RUN_ROOT"))
    parser.add_argument("--steps", type=int, nargs="+", default=[5, 25, 50, 100])
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(90730)
    pipe = YuE2Pipeline.from_pretrained(args.model, vae=args.vae, device="cuda",
                                        backend="torch-eager", memory_budget_gib=32,
                                        progress=False)
    base = pipe._load_model().eval()
    policy = PeftModel.from_pretrained(base, args.reference_adapter, is_trainable=False).eval()
    rollouts = reference_rollouts(pipe, args.output, args.max_tokens)
    if pipe._vae is not None:
        pipe._vae.to("cpu")
    policy.to("cuda")
    policy.load_adapter(args.reference_adapter, adapter_name="probe")
    policy.set_adapter("probe")
    policy.eval()
    references = [constrained_log_probs(policy, rollout) for rollout in rollouts]
    policy.set_adapter("default")
    policy.delete_adapter("probe")
    policy.load_adapter(args.reference_adapter, adapter_name="probe")
    policy.set_adapter("probe")
    policy.eval()
    identity_kl = [conditional_kl(reference, constrained_log_probs(policy, rollout)[0])
                   for rollout, (reference, _) in zip(rollouts, references)]
    if max(abs(value) for value in identity_kl) > 1e-5:
        raise AssertionError(f"Same-weight adapter identity KL is nonzero: {identity_kl}")
    policy.set_adapter("default")
    policy.delete_adapter("probe")
    report = {
        "status": "offline_fixed_reference_conditional_kl",
        "reference": "YuE2 one-step source LoRA, not the frozen zero-step base",
        "definition": "mean token D_KL(reference || checkpoint) on three reference-generated held-out trajectories; codec vocabulary plus permitted end token, before top-k/top-p/repetition filtering",
        "not_training_kl": True,
        "seeds": [row["seed"] for row in rollouts],
        "identity_kl": identity_kl,
        "reference_allowed_vocab_mass": [mass for _, mass in references],
        "arms": {},
    }
    for name, root_name in args.arm:
        root = Path(root_name)
        report["arms"][name] = {"1": {"kl": 0.0, "per_prompt": [0.0] * len(rollouts)}}
        for step in args.steps:
            adapter = root / "checkpoints" / f"step_{step:06d}" / "adapter"
            if not adapter.is_dir():
                raise FileNotFoundError(adapter)
            policy.load_adapter(adapter, adapter_name="probe")
            policy.set_adapter("probe")
            policy.eval()
            values = []
            allowed_mass = []
            for rollout, (ref_logp, _) in zip(rollouts, references):
                checkpoint_logp, mass = constrained_log_probs(policy, rollout)
                kl = conditional_kl(ref_logp, checkpoint_logp)
                values.append(kl)
                allowed_mass.append(mass)
            report["arms"][name][str(step)] = {
                "kl": sum(values) / len(values), "per_prompt": values,
                "checkpoint_allowed_vocab_mass": allowed_mass,
            }
            policy.set_adapter("default")
            policy.delete_adapter("probe")
            write_json(args.output / "kl.json", report)
            print(f"{name} step={step} KL={report['arms'][name][str(step)]['kl']:.6f}", flush=True)


if __name__ == "__main__":
    main()
