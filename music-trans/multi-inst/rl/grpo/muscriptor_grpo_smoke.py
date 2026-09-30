"""One-input MuScriptor-medium token-head GRPO feasibility smoke.

Uses a preauthorized local safetensors checkpoint. Keeps the transformer and
audio conditioner frozen; only the output token projection is updated. This
script deliberately does not claim generalization from a single example.
"""

import argparse
import io
import json
import random
from pathlib import Path

import numpy as np
import pretty_midi
import torch
import torch.nn.functional as F

from muscriptor.events import ChunkBoundary, decode_model_tokens
from muscriptor.transcription_model import TranscriptionModel, instrument_group_from_names
from note_metrics import Note, score


def notes_from_midi(source, duration):
    midi = pretty_midi.PrettyMIDI(source if isinstance(source, str) else io.BytesIO(source))
    return [Note(int(note.pitch), max(0.0, float(note.start)),
                 min(duration, float(note.end)), "S")
            for instrument in midi.instruments if not instrument.is_drum
            for note in instrument.notes
            if note.start < duration and min(duration, note.end) > max(0.0, note.start)]


def metrics(reference, midi_bytes, duration):
    values = score(reference, notes_from_midi(midi_bytes, duration))
    return {key: values[key]["f1"] for key in ("frame", "onset", "offset")}


def decode_candidate(transcriber, token_ids):
    tokens = [int(token) for token in token_ids if int(token) != transcriber._tokenizer.eos_id]
    events = decode_model_tokens(
        iter([ChunkBoundary(0.0, None), *tokens]),
        transcriber._tokenizer._vocab,
        transcriber._instrument_for_program,
        frame_rate=transcriber._tokenizer.frame_rate,
    )
    return transcriber.events_to_midi_bytes(events)


def voice_forbidden_tokens(transcriber):
    invalid_ids = range(len(transcriber._tokenizer._vocab),
                        transcriber._model.linear.out_features)
    forbidden_ids = sorted(set(transcriber._tokenizer.forbidden_token_ids(["voice"])) |
                           set(invalid_ids))
    return torch.tensor(forbidden_ids, dtype=torch.long, device=transcriber._device)


def valid_token_index(transcriber, forbidden):
    card = transcriber._model.linear.out_features
    valid = torch.ones(card, dtype=torch.bool, device=transcriber._device)
    valid[forbidden] = False
    selected = torch.arange(card, device=transcriber._device)[valid]
    lookup = torch.full((card,), -1, dtype=torch.long, device=transcriber._device)
    lookup[selected] = torch.arange(len(selected), device=transcriber._device)
    return valid, lookup


def token_hidden(transcriber, tokens, conditions):
    model = transcriber._model
    context = torch.tensor([model.initial_token_id, *tokens[:-1]],
                           dtype=torch.long, device=transcriber._device)[None]
    captured = []

    def capture_input(_module, inputs):
        captured.append(inputs[0].detach().clone())

    hook = model.linear.register_forward_pre_hook(capture_input)
    try:
        with torch.no_grad(), model.autocast:
            model(context, conditions, first_step=True)
    finally:
        hook.remove()
    if len(captured) != 1:
        raise RuntimeError("Could not capture the frozen decoder features")
    return captured[0][0]


def token_log_probs(hidden, weight, tokens, valid, lookup, autocast):
    targets = torch.tensor(tokens, dtype=torch.long, device=hidden.device)
    positions = lookup[targets]
    if (positions < 0).any():
        raise ValueError("Sampled a token that the replay policy masks")
    with autocast:
        logits = F.linear(hidden, weight)[:, valid].float()
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs, log_probs.gather(1, positions[:, None]).squeeze(1)


def greedy_midi(transcriber, condition, forbidden, max_tokens):
    tokens = []
    with torch.inference_mode():
        for item in transcriber._model.generate(
            conditions=[condition], max_gen_len=max_tokens, use_sampling=False,
            cfg_coef=1.0, early_stop_on_token=transcriber._tokenizer.eos_id,
            forbidden_tokens=forbidden,
        ):
            token = int(item[0])
            tokens.append(token)
            if token == transcriber._tokenizer.eos_id:
                break
    return decode_candidate(transcriber, tokens), len(tokens)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()
    if args.steps < 1 or args.group_size < 2 or args.max_tokens < 2:
        parser.error("Need positive steps, group size >=2 and max tokens >=2")
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    transcriber = TranscriptionModel.load_model(args.checkpoint, device="cuda")
    model = transcriber._model
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.linear.parameters():
        parameter.requires_grad_(True)
    reference_weight = model.linear.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.linear.parameters(), lr=1e-5)

    duration = 5.0
    reference = notes_from_midi(str(args.reference), duration)
    wav = transcriber._load_wav(args.input, None)[:, : int(16000 * duration)]
    if wav.shape[-1] < int(16000 * duration):
        wav = F.pad(wav, (0, int(16000 * duration) - wav.shape[-1]))
    condition = transcriber._build_conditions(wav, instrument_group_from_names(["voice"]))[0]
    forbidden = voice_forbidden_tokens(transcriber)
    valid, lookup = valid_token_index(transcriber, forbidden)
    with torch.inference_mode():
        encoded = model.condition_provider(model.condition_provider.tokenize([condition]))
    before_midi, before_tokens = greedy_midi(transcriber, condition, forbidden, args.max_tokens)
    (args.output / "before.mid").write_bytes(before_midi)
    before = metrics(reference, before_midi, duration)
    print("before", before, "tokens", before_tokens, flush=True)

    trace = []
    for step in range(1, args.steps + 1):
        groups = [[] for _ in range(args.group_size)]
        with torch.inference_mode():
            conditions = [condition] * args.group_size
            for item in model.generate(
                conditions=conditions, max_gen_len=args.max_tokens,
                use_sampling=True, temp=1.0, top_k=0, top_p=0.0,
                cfg_coef=1.0, early_stop_on_token=transcriber._tokenizer.eos_id,
                forbidden_tokens=forbidden,
            ):
                for index, token in enumerate(item.tolist()):
                    if not groups[index] or groups[index][-1] != transcriber._tokenizer.eos_id:
                        groups[index].append(int(token))
        samples = []
        for tokens in groups:
            midi_bytes = decode_candidate(transcriber, tokens)
            measured = metrics(reference, midi_bytes, duration)
            reward = 0.45 * measured["onset"] + 0.25 * measured["offset"] + 0.30 * measured["frame"]
            hidden = token_hidden(transcriber, tokens, encoded)
            with torch.inference_mode():
                reference_log_probs, _ = token_log_probs(
                    hidden, reference_weight, tokens, valid, lookup, model.autocast)
                _, old_log_probs = token_log_probs(
                    hidden, model.linear.weight, tokens, valid, lookup, model.autocast)
            samples.append((tokens, measured, reward, hidden, reference_log_probs.detach(),
                            old_log_probs.detach()))
        rewards = torch.tensor([sample[2] for sample in samples], device=transcriber._device)
        spread = float(rewards.std(unbiased=False))
        row = {"step": step, "rewards": rewards.tolist(), "spread": spread,
               "updated": False, "token_counts": [len(sample[0]) for sample in samples]}
        if spread > 1e-7:
            advantages = (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(1e-7)
            epoch_kl = []
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                losses = []
                kl_values = []
                for advantage, (tokens, _, _, hidden, reference_distribution, old) in zip(advantages, samples):
                    current_distribution, current = token_log_probs(
                        hidden, model.linear.weight, tokens, valid, lookup, model.autocast)
                    ratio = (current - old).clamp(-20, 20).exp()
                    objective = torch.minimum(ratio * advantage,
                                              ratio.clamp(0.8, 1.2) * advantage)
                    reference_prob = reference_distribution.exp()
                    kl = (reference_prob * (reference_distribution - current_distribution)).sum(-1)
                    kl_values.append(kl.mean().detach())
                    losses.append(-objective.mean() + 0.01 * kl.mean())
                torch.stack(losses).mean().backward()
                torch.nn.utils.clip_grad_norm_(model.linear.parameters(), 1.0)
                optimizer.step()
                epoch_kl.append(float(torch.stack(kl_values).mean()))
            row["updated"] = True
            row["reference_kl_by_epoch"] = epoch_kl
        trace.append(row)
        print("step", step, "spread", round(spread, 5), "updated", row["updated"], flush=True)

    after_midi, after_tokens = greedy_midi(transcriber, condition, forbidden, args.max_tokens)
    (args.output / "after.mid").write_bytes(after_midi)
    after = metrics(reference, after_midi, duration)
    receipt = {
        "status": "single_public_song_muscriptor_medium_grpo_smoke_not_heldout",
        "checkpoint_variant": "muscriptor-medium",
        "policy": "frozen_acoustic_conditioner_and_transformer_trainable_token_projection",
        "seed": args.seed, "steps_requested": args.steps,
        "steps_updated": sum(row["updated"] for row in trace),
        "group_size": args.group_size, "max_tokens": args.max_tokens,
        "duration_seconds": duration, "reference_note_count": len(reference),
        "reward": "0.45 onset F1 + 0.25 onset+offset F1 + 0.30 frame F1, pitch-only",
        "reference": "Frozen initial token head; KL(reference || current) over allowed tokens",
        "before": {"metrics": before, "token_count": before_tokens},
        "after": {"metrics": after, "token_count": after_tokens},
        "trace": trace,
        "caveat": "One training song is also the replay song; no held-out or SATB-track-aware claim.",
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    torch.save(model.linear.state_dict(), args.output / "token_head.pt")
    print("after", after, "tokens", after_tokens, flush=True)


if __name__ == "__main__":
    main()
