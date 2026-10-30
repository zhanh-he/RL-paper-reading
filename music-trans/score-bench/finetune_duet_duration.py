"""Small-data SFT / note-action GRPO of DuET's duration head on real ASAP pairs.

The encoder/decoder and baseline-generated history stay frozen while caching hidden
states. GRPO samples singleton duration actions within each note context; it is
not an autoregressive sequence-policy GRPO implementation.
"""
import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import mido
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from duet import DuET
from duet.features import build_window
from duet.midi import read_performance_midi


def score_notes(filename):
    midi = mido.MidiFile(filename)
    ticks, seconds, tempo = 0, 0., 500000
    active, notes = defaultdict(list), []
    for msg in mido.merge_tracks(midi.tracks):
        seconds += mido.tick2second(msg.time, midi.ticks_per_beat, tempo)
        ticks += msg.time
        if msg.type == "set_tempo":
            tempo = msg.tempo
        if msg.type == "note_on" and msg.velocity > 0:
            active[(msg.channel, msg.note)].append((ticks, seconds))
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            key = (msg.channel, msg.note)
            if active[key]:
                start_tick, start_sec = active[key].pop(0)
                notes.append((msg.note, start_sec, (ticks-start_tick)/midi.ticks_per_beat))
    return np.asarray(notes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--method", choices=["sft", "grpo"], default="sft")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(20261030); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    pipeline = DuET.from_pretrained(args.weights, device="cuda")
    model, codec = pipeline.model, pipeline.model.codec
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    cached = {"train": [], "validation": []}
    manifest = json.loads((args.data / "manifest.json").read_text())
    alignment = []
    for sample in manifest["samples"]:
        split = sample["split"]
        if split not in cached:
            continue
        files = {k: args.data / v["path"] for k, v in sample["files"].items()}
        data = read_performance_midi(files["midi_performance"])
        perf_beats = np.loadtxt(files["performance_annotations"], usecols=0, ndmin=1)
        score_beats = np.loadtxt(files["midi_score_annotations"], usecols=0, ndmin=1)
        if len(perf_beats) != len(score_beats):
            alignment.append({"id": sample["id"], "status": "skipped", "reason": "beat-count mismatch"}); continue
        batch, ids, _ = build_window(data, 0.)
        # First 128 notes from the first official 32s region; no test data used.
        ids = ids[:128]
        batch["query_note_slots"] = batch["query_note_slots"][:, :len(ids)]
        batch = {k: v.to("cuda") for k, v in batch.items()}
        target_score = score_notes(files["midi_score"])
        warped = np.interp(data["ref_onset"][ids], perf_beats, score_beats)
        labels = np.full(len(ids), -100, dtype=np.int64)
        for pitch in np.unique(data["ref_pitch"][ids]):
            slots = np.flatnonzero(data["ref_pitch"][ids] == pitch)
            targets = np.flatnonzero(target_score[:, 0] == pitch)
            if not len(targets): continue
            cost = np.abs(warped[slots, None] - target_score[targets, 1][None])
            rows, columns = linear_sum_assignment(cost)
            for row, column in zip(rows, columns):
                if cost[row, column] > .10: continue
                duration = target_score[targets[column], 2] * codec.ppq
                nearest = int(np.argmin(np.abs(codec.tick_values["duration"] - duration)))
                if abs(codec.tick_values["duration"][nearest] - duration) <= 1:
                    labels[slots[row]] = nearest
        keep = labels >= 0
        if keep.sum() < 16:
            alignment.append({"id": sample["id"], "status": "skipped", "reason": "fewer than 16 aligned notes"}); continue
        with torch.no_grad():
            predictions = model.generate_rows(batch)
            previous = torch.cat([torch.zeros_like(predictions[:1]), predictions[:-1]], dim=0)[None]
            known = torch.ones_like(previous, dtype=torch.bool); known[:, 0] = False
            batch.update(decoder_input_ids=previous, decoder_input_field_mask=known,
                decoder_attention_mask=torch.ones(1, len(ids), dtype=torch.bool, device="cuda"),
                decoder_context=torch.zeros(1, len(ids), dtype=torch.bool, device="cuda"))
            memory, valid, _ = model.encode(batch)
            _, hidden = model.decode_aligned(batch, memory, valid, return_hidden=True)
            cached[split].append((hidden[0, keep].detach(), torch.tensor(labels[keep], device="cuda")))
        alignment.append({"id": sample["id"], "status": "cached", "input_notes": len(ids), "aligned_duration_labels": int(keep.sum())})
        print(json.dumps(alignment[-1]), flush=True)
    for split in cached:
        if not cached[split]: raise RuntimeError(f"No aligned {split} examples")
    head = model.heads[3]
    original = {k: v.detach().clone() for k, v in head.state_dict().items()}
    for parameter in head.parameters(): parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=.01)
    def evaluate():
        with torch.no_grad():
            return {split: {"n_notes": sum(len(y) for x,y in items),
                "accuracy": sum(int((head(x).argmax(-1) == y).sum()) for x,y in items) / sum(len(y) for x,y in items),
                "cross_entropy": sum(float(torch.nn.functional.cross_entropy(head(x),y,reduction="sum")) for x,y in items) / sum(len(y) for x,y in items)} for split,items in cached.items()}
    before = evaluate(); trace = []
    for step in range(1, args.steps + 1):
        x, y = cached["train"][(step-1) % len(cached["train"])]
        optimizer.zero_grad()
        logits = head(x)
        ce = torch.nn.functional.cross_entropy(logits,y)
        record = {"step": step, "train_cross_entropy": float(ce.detach())}
        if args.method == "sft":
            loss, effective = ce, True
        else:
            distribution = torch.distributions.Categorical(logits=logits)
            actions = distribution.sample((8,))
            rewards = (actions == y[None]).float()
            std = rewards.std(dim=0, unbiased=False)
            advantage = (rewards-rewards.mean(dim=0,keepdim=True)) / (std[None]+1e-8)
            log_prob = distribution.log_prob(actions)
            ratio = (log_prob-log_prob.detach()).exp()
            policy_loss = -torch.minimum(ratio*advantage, ratio.clamp(.8,1.2)*advantage).mean()
            with torch.no_grad():
                reference = torch.nn.functional.linear(x, original["weight"], original["bias"]).log_softmax(-1)
            kl = (logits.softmax(-1)*(logits.log_softmax(-1)-reference)).sum(-1).mean()
            loss = policy_loss + .01*kl
            effective = bool((std > 0).any())
            record.update(reward=float(rewards.mean()), kl=float(kl.detach()), variable_groups=int((std>0).sum()), group_size=8)
        if effective:
            loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.); optimizer.step()
        record["optimizer_update"] = effective
        trace.append(record)
    after = evaluate()
    torch.save(head.state_dict(), args.output / "duration-head.pt")
    receipt = {"method": "Supervised duration-head adaptation with frozen baseline-history hidden states" if args.method == "sft" else "Note-action duration-head GRPO on frozen baseline-history hidden states", "optimizer_steps": sum(row["optimizer_update"] for row in trace),
        "learning_rate": 1e-4, "seed": 20261030, "trainable_parameters": sum(p.numel() for p in head.parameters()),
        "source_checkpoint": pipeline.metadata, "before": before, "after": after, "trace": trace, "alignment": alignment,
        "head_l2_change": sum(float((head.state_dict()[k]-v).square().sum()) for k,v in original.items())**.5,
        "limitations": "Own confidence-filtered beat-warp/pitch matching; only first <=128 notes per train/valid work. Cached-context duration-head adaptation; GRPO uses 8 categorical singleton actions per note, exact forward KL to frozen duration head, beta .01, one optimizer update per rollout. This is not full-model SFT or autoregressive sequence GRPO. Full-piece downstream evaluation is separate."}
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    # Evaluate the altered score head through the original complete-piece pipeline.
    downstream = []
    for sample in manifest["samples"]:
        if sample["split"] != "test": continue
        start = time.monotonic()
        try:
            report = pipeline.transcribe(args.data / sample["files"]["midi_performance"]["path"], args.output / sample["id"])
            downstream.append({"id": sample["id"], "status": "complete", "seconds": time.monotonic()-start, "report": report})
        except Exception as error:
            downstream.append({"id": sample["id"], "status": "failed", "error": repr(error)})
        receipt["downstream"] = downstream
        (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__": main()
