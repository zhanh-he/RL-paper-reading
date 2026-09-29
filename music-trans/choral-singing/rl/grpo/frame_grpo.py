"""Small-data GRPO pilot for ChoralStream's onset/frame/offset policy head.

This trains the frame policy only. It does not update the autoregressive MIDI
decoder or claim a SATB track-note improvement.
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


HEADS = ("onset", "frame", "offset")


def score_bits(bits, target):
    bits = bits.bool()
    target = target.bool()
    tp = (bits & target).sum(dim=(-2, -1)).float()
    fp = (bits & ~target).sum(dim=(-2, -1)).float()
    fn = (~bits & target).sum(dim=(-2, -1)).float()
    denom = 2 * tp + fp + fn
    return torch.where(denom > 0, 2 * tp / denom.clamp_min(1), torch.ones_like(denom))


def component_scores(bits, target):
    return torch.stack([score_bits(bits[..., i, :], target[..., i, :]) for i in range(3)], dim=-1)


def scalar_reward(components, arm):
    if arm == "combined":
        return (components * components.new_tensor([0.25, 0.5, 0.25])).sum(dim=-1)
    return components[..., HEADS.index(arm)]


def bernoulli_kl(logits, reference_logits):
    p = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
    q = torch.sigmoid(reference_logits).clamp(1e-6, 1 - 1e-6)
    return (p * (p.log() - q.log()) + (1 - p) * ((1 - p).log() - (1 - q).log())).mean()


def original_ids(manifest, limit):
    ids = json.loads(Path(manifest).read_text())
    seen = set()
    selected = []
    for fid in ids:
        source = fid.split("_psm")[0].split("_psp")[0]
        if source in seen or fid != source:
            continue
        seen.add(source)
        selected.append(fid)
        if limit and len(selected) == limit:
            break
    return selected


def prepare_split(dataset_class, root, split, limit, output_dir, segment_index):
    ids = original_ids(root / f"{split}.json", limit)
    manifest = output_dir / f"{split}_selected.json"
    manifest.write_text(json.dumps(ids, indent=2) + "\n")
    dataset = dataset_class(
        root / "mel", root / "note", manifest, seg_len=320, shuffle=False, mel_cache_size=1
    )
    chosen = {}
    for dataset_index, (entry_index, seg_index) in enumerate(dataset._index_map):
        fid = dataset.entries[entry_index]["fid"]
        if fid not in chosen or (seg_index == segment_index):
            chosen[fid] = dataset_index
    return [dataset[chosen[fid]] for fid in ids if fid in chosen]


@torch.no_grad()
def encode_samples(model, samples, device):
    model.eval()
    encoded = []
    for sample in samples:
        mel = sample["mel"].unsqueeze(0).to(device)
        features = model._encode_mel(mel).detach()
        target = torch.stack([sample[key] for key in HEADS], dim=1).to(device)
        encoded.append((features, target, sample["fid"]))
    return encoded


@torch.no_grad()
def evaluate(head, encoded):
    totals = torch.zeros(3, 3, dtype=torch.float64)
    for features, target, _ in encoded:
        pred = (head(features)[0].view_as(target) > 0)
        truth = target.bool()
        for i in range(3):
            totals[i, 0] += (pred[:, i] & truth[:, i]).sum().item()
            totals[i, 1] += (pred[:, i] & ~truth[:, i]).sum().item()
            totals[i, 2] += (~pred[:, i] & truth[:, i]).sum().item()
    output = {}
    for i, name in enumerate(HEADS):
        tp, fp, fn = totals[i].tolist()
        output[name] = {
            "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "tp": int(tp), "fp": int(fp), "fn": int(fn),
        }
    return output


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = Path(args.choralstream).resolve()
    sys.path.insert(0, str(source))
    from dataset.choral_dataset import ChoralMelDataset
    from model import ChoralStreamModel

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    config = json.loads(Path(args.config).read_text())
    model = ChoralStreamModel(
        n_layers=config["n_layers"], seg_len=config["seg_len"],
        enable_encoder=config["enable_encoder"], prob_model=config["prob_model"],
        use_voice_queries=config["use_voice_queries"],
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.frame_prj.parameters():
        parameter.requires_grad_(True)
    model.eval()
    root = Path(args.data).resolve()
    train = encode_samples(model, prepare_split(ChoralMelDataset, root, "train", args.train_songs, output, args.segment_index), device)
    valid = encode_samples(model, prepare_split(ChoralMelDataset, root, "valid", args.valid_songs, output, args.segment_index), device)
    test = encode_samples(model, prepare_split(ChoralMelDataset, root, "test", args.test_songs, output, args.segment_index), device)
    if not train or not valid or not test:
        raise ValueError("Each split needs at least one usable original song")
    train_ids = {fid for _, _, fid in train}
    valid_ids = {fid for _, _, fid in valid}
    test_ids = {fid for _, _, fid in test}
    if (train_ids & valid_ids) or (train_ids & test_ids) or (valid_ids & test_ids):
        raise ValueError("Source IDs overlap across train, valid, and test")

    with torch.no_grad():
        reference_logits = {
            fid: model.frame_prj(features)[0].view_as(target).detach().clone()
            for features, target, fid in train
        }
        positives = torch.stack([target.sum(dim=(0, 2)) for _, target, _ in train]).sum(dim=0)
        total = sum(target.shape[0] * target.shape[2] for _, target, _ in train)
        pos_weight = ((total - positives) / positives.clamp_min(1)).clamp(max=100).view(1, 3, 1)
    before_valid = evaluate(model.frame_prj, valid)
    before_test = evaluate(model.frame_prj, test)
    optimizer = torch.optim.AdamW(model.frame_prj.parameters(), lr=args.lr)
    trace = []
    for epoch in range(args.epochs):
        order = list(range(len(train)))
        random.shuffle(order)
        for sample_index in order:
            features, target, fid = train[sample_index]
            if args.algorithm == "bce":
                for update_pass in range(args.update_epochs):
                    new_logits = model.frame_prj(features)[0].view_as(target)
                    loss = F.binary_cross_entropy_with_logits(new_logits, target, pos_weight=pos_weight)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.frame_prj.parameters(), 1.0)
                    optimizer.step()
                    trace.append({
                        "epoch": epoch, "update_pass": update_pass, "source": fid,
                        "skipped": False, "loss": loss.item(),
                    })
                continue
            with torch.no_grad():
                old_logits = model.frame_prj(features)[0].view_as(target).detach()
                old_probs = torch.sigmoid(old_logits)
                actions = torch.bernoulli(old_probs.expand(args.group_size, *old_probs.shape))
                components = component_scores(actions, target)
                rewards = scalar_reward(components, args.arm)
                spread = rewards.std(unbiased=False)
                if spread < args.min_group_std:
                    trace.append({"epoch": epoch, "source": fid, "skipped": True, "group_std": spread.item()})
                    continue
                advantage = ((rewards - rewards.mean()) / spread).detach()
                old_logp = -F.binary_cross_entropy_with_logits(
                    old_logits.expand_as(actions), actions, reduction="none"
                )
            for update_pass in range(args.update_epochs):
                new_logits = model.frame_prj(features)[0].view_as(target)
                new_logp = -F.binary_cross_entropy_with_logits(
                    new_logits.expand_as(actions), actions, reduction="none"
                )
                ratio = (new_logp - old_logp).clamp(-10, 10).exp()
                unclipped = ratio * advantage[:, None, None, None]
                clipped = ratio.clamp(1 - args.clip_eps, 1 + args.clip_eps) * advantage[:, None, None, None]
                loss = -torch.minimum(unclipped, clipped).mean()
                loss = loss + args.kl_beta * bernoulli_kl(new_logits, reference_logits[fid])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.frame_prj.parameters(), 1.0)
                optimizer.step()
                trace.append({
                    "epoch": epoch, "update_pass": update_pass, "source": fid, "skipped": False,
                    "reward_mean": rewards.mean().item(), "group_std": spread.item(),
                    "loss": loss.item(),
                    "clip_fraction": ((ratio < 1 - args.clip_eps) | (ratio > 1 + args.clip_eps)).float().mean().item(),
                })

    after_valid = evaluate(model.frame_prj, valid)
    after_test = evaluate(model.frame_prj, test)
    torch.save(model.frame_prj.state_dict(), output / "frame_head.pt")
    (output / "trace.jsonl").write_text("\n".join(json.dumps(row) for row in trace) + "\n")
    receipt = {
        "status": "local_online_grpo_frame_policy_pilot" if args.algorithm == "grpo" else "local_bce_frame_policy_control",
        "scope": "ChoralStream frame/onset/offset head only; not MIDI event decoding or SATB track-note",
        "algorithm": args.algorithm, "arm": args.arm if args.algorithm == "grpo" else "weighted_bce_all_heads",
        "seed": args.seed, "epochs": args.epochs,
        "group_size": args.group_size, "update_epochs": args.update_epochs,
        "train_original_songs": len(train),
        "valid_original_songs": len(valid), "test_original_songs": len(test),
        "segment_index": args.segment_index, "clip_eps": args.clip_eps,
        "lr": args.lr, "kl_beta": args.kl_beta,
        "bce_pos_weight": pos_weight.flatten().tolist() if args.algorithm == "bce" else None,
        "train_steps": sum(not row["skipped"] for row in trace),
        "skipped_groups": sum(row["skipped"] for row in trace),
        "mean_clip_fraction": sum(row.get("clip_fraction", 0.0) for row in trace) / max(len(trace), 1),
        "before_valid": before_valid, "after_valid": after_valid,
        "before_test": before_test, "after_test": after_test,
        "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--choralstream", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--arm", choices=(*HEADS, "combined"), default="combined")
    parser.add_argument("--algorithm", choices=("grpo", "bce"), default="grpo")
    parser.add_argument("--train-songs", type=int, default=32)
    parser.add_argument("--valid-songs", type=int, default=8)
    parser.add_argument("--test-songs", type=int, default=8)
    parser.add_argument("--segment-index", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--kl-beta", type=float, default=0.01)
    parser.add_argument("--min-group-std", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
