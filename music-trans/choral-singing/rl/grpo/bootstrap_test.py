"""Paired song-level bootstrap for saved ChoralStream frame-head runs."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from frame_grpo import HEADS, encode_samples, prepare_split


def counts(head, encoded):
    rows = []
    with torch.no_grad():
        for features, target, _ in encoded:
            pred = head(features)[0].view_as(target) > 0
            truth = target.bool()
            per_head = []
            for i in range(3):
                per_head.append([
                    (pred[:, i] & truth[:, i]).sum().item(),
                    (pred[:, i] & ~truth[:, i]).sum().item(),
                    (~pred[:, i] & truth[:, i]).sum().item(),
                ])
            rows.append(per_head)
    return np.asarray(rows, dtype=np.int64)


def f1(rows):
    totals = rows.sum(axis=0)
    return 2 * totals[:, 0] / np.maximum(2 * totals[:, 0] + totals[:, 1] + totals[:, 2], 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--choralstream", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--head", action="append", required=True, help="name=frame_head.pt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()

    source = Path(args.choralstream).resolve()
    sys.path.insert(0, str(source))
    from dataset.choral_dataset import ChoralMelDataset
    from model import ChoralStreamModel
    from constants import HOP_LENGTH, SAMPLE_RATE

    config = json.loads(Path(args.config).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChoralStreamModel(
        n_layers=config["n_layers"], seg_len=config["seg_len"],
        enable_encoder=config["enable_encoder"], prob_model=config["prob_model"],
        use_voice_queries=config["use_voice_queries"],
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    selected = json.loads((Path(args.manifest_dir) / "test_selected.json").read_text())
    samples = prepare_split(
        ChoralMelDataset, Path(args.data).resolve(), "test", len(selected),
        Path(args.manifest_dir).resolve(), segment_index=2,
    )
    encoded = encode_samples(model, samples, device)
    baseline = counts(model.frame_prj, encoded)
    rng = np.random.default_rng(args.seed)
    draws = rng.integers(0, len(encoded), size=(args.draws, len(encoded)))
    result = {
        "status": "paired_song_bootstrap",
        "songs": len(encoded), "clip_seconds": 320 * HOP_LENGTH / SAMPLE_RATE,
        "draws": args.draws, "seed": args.seed,
        "baseline_f1": dict(zip(HEADS, f1(baseline).tolist())),
        "runs": {},
    }
    for item in args.head:
        name, path = item.split("=", 1)
        model.frame_prj.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        variant = counts(model.frame_prj, encoded)
        delta = np.stack([f1(variant[index]) - f1(baseline[index]) for index in draws])
        result["runs"][name] = {
            "f1": dict(zip(HEADS, f1(variant).tolist())),
            "delta_f1": dict(zip(HEADS, (f1(variant) - f1(baseline)).tolist())),
            "delta_ci95": {
                head: np.quantile(delta[:, i], [0.025, 0.975]).tolist()
                for i, head in enumerate(HEADS)
            },
        }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
