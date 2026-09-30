"""Small on-policy GRPO pilot on ChoralStream's MIDI pitch/voice event heads.

This is a 5.12-second segment experiment. It deliberately updates the heads
used by decode_events, unlike the earlier frame-head pilot. The acoustic
encoder and frame head remain frozen.
"""

import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from frame_grpo import prepare_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rewards"))
from note_metrics import Note, VOICES, score
from satb_reward import reward


START_SIGMA = 0.025
DURATION_SIGMA = 0.04


def notes_from_target(sample, min_midi):
    seconds = float(sample["end_time"] - sample["begin_time"])
    return [
        Note(int(p) + min_midi, float(s) * seconds, float(s + d) * seconds, VOICES[int(v)])
        for p, s, d, v in zip(sample["pitch"][1:-1], sample["start"][1:-1],
                              sample["dur"][1:-1], sample["voice"][1:-1])
        if 0 <= int(v) < 4 and d > 0
    ]


def notes_from_actions(pitch, start, dur, voice, min_midi, max_midi, seconds):
    output = []
    for p, s, d, v in zip(pitch, start, dur, voice):
        p, v = int(p), int(v)
        if p + min_midi > max_midi or v not in range(4):
            continue
        onset = max(0.0, float(s) * seconds)
        offset = min(seconds, float(s + d) * seconds)
        if onset < seconds and offset - onset >= 0.05:
            output.append(Note(p + min_midi, onset, offset, VOICES[v]))
    return output


@torch.no_grad()
def sample_group(model, mel, group_size, max_tokens, min_midi, max_midi):
    from transformer.Models import get_pad_mask, get_subsequent_mask
    from constants import INI_IDX, EOS_IDX, PAD_IDX, MRK_IDX, VOICE_PAD_IDX

    encoded = model._encode_mel(mel).expand(group_size, -1, -1)
    device = mel.device
    pitch = torch.full((group_size, 1), INI_IDX, device=device, dtype=torch.long)
    start = torch.zeros((group_size, 1), device=device)
    dur = torch.zeros((group_size, 1), device=device)
    voice = torch.full((group_size, 1), VOICE_PAD_IDX, device=device, dtype=torch.long)
    active = torch.ones(group_size, device=device, dtype=torch.bool)
    masks = []
    for _ in range(max_tokens):
        previous_voice = torch.roll(voice, 1, dims=1)
        previous_voice[:, 0] = VOICE_PAD_IDX
        decoder_input = model._build_decoder_input(pitch, start, dur, previous_voice)
        mask = get_pad_mask(pitch, PAD_IDX) & get_subsequent_mask(pitch)
        decoder, *_ = model.decoder(decoder_input, mask, encoded)
        latest = decoder[:, -1]
        pitch_logits = model.trg_pitch_prj(latest).float()
        supported = model.get_valid_voice_mask_for_pitch(
            torch.arange(pitch_logits.shape[-1], device=device)
        ).any(dim=-1)
        pitch_logits[:, ~supported] = -1e9
        pitch_logits[:, [INI_IDX, MRK_IDX, PAD_IDX]] = -1e9
        new_pitch = torch.distributions.Categorical(logits=pitch_logits).sample()
        valid_note = active & (new_pitch != EOS_IDX)
        valid_voice = model.get_valid_voice_mask_for_pitch(new_pitch)
        voice_logits = model.trg_voice_prj(latest).float()[:, :4]
        voice_logits = voice_logits.masked_fill(~valid_voice, -1e9)
        new_voice = torch.distributions.Categorical(logits=voice_logits).sample()
        start_mean = torch.sigmoid(model.trg_start_prj(latest)).squeeze(-1)
        dur_mean = torch.sigmoid(model.trg_dur_prj(latest)).squeeze(-1)
        new_start = torch.distributions.Normal(start_mean, START_SIGMA).sample()
        new_dur = torch.distributions.Normal(dur_mean, DURATION_SIGMA).sample()
        new_pitch = torch.where(active, new_pitch, PAD_IDX)
        new_voice = torch.where(valid_note, new_voice, VOICE_PAD_IDX)
        new_start = torch.where(valid_note, new_start, 0.0)
        new_dur = torch.where(valid_note, new_dur, 0.0)
        masks.append(active)
        pitch = torch.cat([pitch, new_pitch[:, None]], dim=1)
        voice = torch.cat([voice, new_voice[:, None]], dim=1)
        start = torch.cat([start, new_start[:, None]], dim=1)
        dur = torch.cat([dur, new_dur[:, None]], dim=1)
        active = valid_note
        if not active.any():
            break
    return pitch, start, dur, voice, torch.stack(masks, dim=1)


def action_logprobs(model, mel, actions):
    from constants import EOS_IDX, INI_IDX, MRK_IDX, PAD_IDX

    pitch, start, dur, voice, mask = actions
    batch = pitch.shape[0]
    logits, start_mean, dur_mean, voice_logits = model(
        mel.expand(batch, -1, -1), pitch[:, :-1], start[:, :-1],
        dur[:, :-1], voice[:, :-1],
    )
    logits = logits.float()
    supported = model.get_valid_voice_mask_for_pitch(
        torch.arange(logits.shape[-1], device=logits.device)
    ).any(dim=-1)
    logits[..., ~supported] = -1e9
    logits[..., [INI_IDX, MRK_IDX, PAD_IDX]] = -1e9
    target_pitch = pitch[:, 1:]
    pitch_lp = F.log_softmax(logits, dim=-1).gather(
        -1, target_pitch.clamp_max(EOS_IDX)[..., None]
    ).squeeze(-1)
    voice_target = voice[:, 1:]
    valid_note = mask & (target_pitch != EOS_IDX)
    valid_voice = model.get_valid_voice_mask_for_pitch(target_pitch)
    voice_logits = voice_logits.float()[..., :4].masked_fill(~valid_voice, -1e9)
    voice_lp = F.log_softmax(voice_logits, dim=-1).gather(
        -1, voice_target.clamp_max(3)[..., None]
    ).squeeze(-1)
    time_lp = (
        torch.distributions.Normal(start_mean.squeeze(-1), START_SIGMA).log_prob(start[:, 1:])
        + torch.distributions.Normal(dur_mean.squeeze(-1), DURATION_SIGMA).log_prob(dur[:, 1:])
    )
    return pitch_lp + torch.where(valid_note, voice_lp + time_lp, 0.0)


@torch.no_grad()
def evaluate(model, samples, min_midi, max_midi):
    rows = []
    model.eval()
    for sample in samples:
        mel = sample["mel"].unsqueeze(0).to(next(model.parameters()).device)
        pitch, start, dur, voice = model.decode_events(mel, beam_size=1)
        seconds = float(sample["end_time"] - sample["begin_time"])
        estimate = notes_from_actions(pitch[0], start[0], dur[0], voice[0],
                                      min_midi, max_midi, seconds)
        reference = notes_from_target(sample, min_midi)
        rows.append({"source": sample["fid"], "scores": score(reference, estimate),
                     "estimate_notes": len(estimate), "reference_notes": len(reference)})
    return rows


def aggregate(rows):
    keys = ("frame", "track_frame", "onset", "offset", "track_onset", "track_note")
    sums = {key: [0, 0, 0] for key in keys}
    for row in rows:
        for key in keys:
            value = row["scores"][key]
            for i, component in enumerate(("tp", "fp", "fn")):
                sums[key][i] += value[component]
    overall = {
        key: {"precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
              "f1": 2 * tp / max(2 * tp + fp + fn, 1), "tp": tp, "fp": fp, "fn": fn}
        for key, (tp, fp, fn) in sums.items()
    }
    per_voice = {}
    for voice in VOICES:
        per_voice[voice] = {}
        for metric in ("frame", "onset", "onset_offset"):
            tp = sum(row["scores"]["per_voice"][voice][metric]["tp"] for row in rows)
            fp = sum(row["scores"]["per_voice"][voice][metric]["fp"] for row in rows)
            fn = sum(row["scores"]["per_voice"][voice][metric]["fn"] for row in rows)
            per_voice[voice][metric] = {
                "precision": tp / max(tp + fp, 1),
                "recall": tp / max(tp + fn, 1),
                "f1": 2 * tp / max(2 * tp + fp + fn, 1),
                "tp": tp, "fp": fp, "fn": fn,
            }
    overall["per_voice"] = per_voice
    overall["macro"] = {
        metric: sum(per_voice[voice][metric]["f1"] for voice in VOICES) / 4
        for metric in ("frame", "onset", "onset_offset")
    }
    overall["va_rate_percent"] = {
        "frame": 100 * overall["macro"]["frame"] / max(overall["frame"]["f1"], 1e-12),
        "onset": 100 * overall["macro"]["onset"] / max(overall["onset"]["f1"], 1e-12),
    }
    return overall


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--choralstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", default="combined", choices=(
        "combined", "onset", "onset_offset", "frame", "coverage", "weak_voice", "precision", "continuity"))
    parser.add_argument("--train-songs", type=int, default=8)
    parser.add_argument("--test-songs", type=int, default=4)
    parser.add_argument("--segment-index", type=int, default=2)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--milestones", type=int, nargs="*", default=[])
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--update-epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--kl-beta", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.choralstream.resolve()))
    from constants import MIN_MIDI, MAX_MIDI
    from dataset.choral_dataset import ChoralMelDataset
    from model import ChoralStreamModel

    config = json.loads(args.config.read_text())
    model = ChoralStreamModel(
        n_layers=config["n_layers"], seg_len=config["seg_len"],
        enable_encoder=config["enable_encoder"], prob_model=config["prob_model"],
        use_voice_queries=config["use_voice_queries"],
    ).cuda().eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for head in (model.trg_pitch_prj, model.trg_voice_prj,
                 model.trg_start_prj, model.trg_dur_prj):
        for parameter in head.parameters():
            parameter.requires_grad_(True)
    reference_model = copy.deepcopy(model).eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)

    train = prepare_split(ChoralMelDataset, args.data, "train", args.train_songs,
                          args.output, args.segment_index)
    test = prepare_split(ChoralMelDataset, args.data, "test", args.test_songs,
                         args.output, args.segment_index)
    if not train or not test or ({s["fid"] for s in train} & {s["fid"] for s in test}):
        raise ValueError("Train/test originals must be nonempty and disjoint")
    before = evaluate(model, test, MIN_MIDI, MAX_MIDI)
    print("before", json.dumps(aggregate(before)), flush=True)
    trained_heads = (model.trg_pitch_prj, model.trg_voice_prj,
                     model.trg_start_prj, model.trg_dur_prj)
    optimizer = torch.optim.AdamW(
        [parameter for head in trained_heads for parameter in head.parameters()], lr=args.lr
    )
    trace = []
    milestones = {}
    for step in range(args.steps):
        sample = train[step % len(train)]
        mel = sample["mel"].unsqueeze(0).cuda()
        target = notes_from_target(sample, MIN_MIDI)
        actions = sample_group(model, mel, args.group_size, args.max_tokens, MIN_MIDI, MAX_MIDI)
        pitch, start, dur, voice, mask = actions
        seconds = float(sample["end_time"] - sample["begin_time"])
        estimates = [notes_from_actions(pitch[i, 1:], start[i, 1:], dur[i, 1:],
                                        voice[i, 1:], MIN_MIDI, MAX_MIDI, seconds)
                     for i in range(args.group_size)]
        rewards = torch.tensor([reward(target, notes, args.arm) for notes in estimates], device="cuda")
        spread = rewards.std(unbiased=False)
        if spread < 1e-6:
            trace.append({"step": step + 1, "skipped": True, "reward": rewards.mean().item()})
        else:
            advantage = (rewards - rewards.mean()) / spread
            with torch.no_grad():
                old_lp = action_logprobs(model, mel, actions)
                ref_lp = action_logprobs(reference_model, mel, actions)
            for _ in range(args.update_epochs):
                new_lp = action_logprobs(model, mel, actions)
                ratio = (new_lp - old_lp).clamp(-10, 10).exp()
                unclipped = ratio * advantage[:, None]
                clipped = ratio.clamp(1 - args.clip_eps, 1 + args.clip_eps) * advantage[:, None]
                policy_loss = -(torch.minimum(unclipped, clipped) * mask).sum() / mask.sum()
                log_delta = (ref_lp - new_lp).clamp(-10, 10)
                kl = ((log_delta.exp() - log_delta - 1) * mask).sum() / mask.sum()
                loss = policy_loss + args.kl_beta * kl
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], 1.0)
                optimizer.step()
            trace.append({"step": step + 1, "source": sample["fid"], "skipped": False,
                          "reward_mean": rewards.mean().item(), "reward_std": spread.item(),
                          "loss": loss.item(), "kl": kl.item(),
                          "notes_per_rollout": [len(notes) for notes in estimates]})
            if (step + 1) % 10 == 0 or step == 0:
                print("step", step + 1, "reward", round(rewards.mean().item(), 4),
                      "spread", round(spread.item(), 4), flush=True)
        if step + 1 in args.milestones:
            evaluated = evaluate(model, test, MIN_MIDI, MAX_MIDI)
            stage = args.output / f"step_{step + 1:06d}"
            stage.mkdir(exist_ok=True)
            torch.save({"pitch": model.trg_pitch_prj.state_dict(),
                        "voice": model.trg_voice_prj.state_dict(),
                        "start": model.trg_start_prj.state_dict(),
                        "duration": model.trg_dur_prj.state_dict()}, stage / "event_heads.pt")
            stage_receipt = {"optimizer_step": step + 1, "updated_steps": sum(not row["skipped"] for row in trace),
                             "reward_arm": args.arm, "test_segments": len(test),
                             "metrics": aggregate(evaluated)}
            (stage / "receipt.json").write_text(json.dumps(stage_receipt, indent=2) + "\n")
            milestones[str(step + 1)] = stage_receipt
            print("milestone", step + 1, json.dumps(stage_receipt["metrics"]["macro"]), flush=True)

    after = evaluate(model, test, MIN_MIDI, MAX_MIDI)
    torch.save({"pitch": model.trg_pitch_prj.state_dict(),
                "voice": model.trg_voice_prj.state_dict(),
                "start": model.trg_start_prj.state_dict(),
                "duration": model.trg_dur_prj.state_dict()}, args.output / "event_heads.pt")
    (args.output / "trace.jsonl").write_text("\n".join(json.dumps(row) for row in trace) + "\n")
    receipt = {
        "status": "measured_event_head_grpo_segment_pilot", "model": "ChoralStream",
        "scope": "pitch/voice/start/duration event heads; frozen encoder and frame head",
        "reward_arm": args.arm, "seed": args.seed, "train_original_songs": len(train),
        "test_original_songs": len(test), "segment_index": args.segment_index,
        "segment_seconds": config["seg_len"] * 256 / 16000,
        "steps_requested": args.steps, "steps_updated": sum(not row["skipped"] for row in trace),
        "group_size": args.group_size, "max_tokens": args.max_tokens,
        "update_epochs": args.update_epochs, "lr": args.lr, "kl_beta": args.kl_beta,
        "start_sigma": START_SIGMA, "duration_sigma": DURATION_SIGMA,
        "before": aggregate(before), "after": aggregate(after),
        "milestones": milestones,
        "per_segment": [{"before": a, "after": b} for a, b in zip(before, after)],
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "caveat": "Small segment pilot; no full-song or listening claim. Test notes are never training rewards.",
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: receipt[k] for k in ("status", "steps_updated", "before", "after")}, indent=2))


if __name__ == "__main__":
    main()
