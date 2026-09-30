"""Paired frozen/GRPO MIDI decoding on disjoint 5-second YouChorale excerpts.

Only anonymous aggregate and paired metrics are written. Source recordings,
song IDs, model weights and MIDI outputs remain on the research machine.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from muscriptor.transcription_model import TranscriptionModel, instrument_group_from_names
from muscriptor_grpo_smoke import greedy_midi, metrics, notes_from_midi, voice_forbidden_tokens


def mean_metrics(rows):
    return {name: float(np.mean([row[name] for row in rows]))
            for name in ("frame", "onset", "offset")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--token-head", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--midi-dir", type=Path, required=True)
    parser.add_argument("--exclude-stem", required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")

    manifest = json.loads(args.manifest.read_text())
    chosen = sorted((row for row in manifest["recordings"]
                     if row["dataset"] == "youchorale" and row["stem"] != args.exclude_stem),
                    key=lambda row: row["stem"])[:args.count]
    if len(chosen) != args.count:
        raise ValueError("Not enough disjoint recordings")
    transcriber = TranscriptionModel.load_model(args.checkpoint, device="cuda")
    model = transcriber._model
    base_head = {key: value.detach().clone() for key, value in model.linear.state_dict().items()}
    trained_head = torch.load(args.token_head, map_location="cpu", weights_only=True)
    forbidden = voice_forbidden_tokens(transcriber)
    instrument_group = instrument_group_from_names(["voice"])

    pairs = []
    for row in chosen:
        with h5py.File(row["path"]) as source:
            samples = source["waveform"][:80000].astype(np.float32) / 32768.0
        if len(samples) < 80000:
            samples = np.pad(samples, (0, 80000 - len(samples)))
        wav = torch.from_numpy(samples).to(transcriber._device)[None]
        condition = transcriber._build_conditions(wav, instrument_group)[0]
        reference = notes_from_midi(str(args.midi_dir / f'{row["stem"]}.mid'), 5.0)
        model.linear.load_state_dict(base_head)
        before_midi, before_tokens = greedy_midi(transcriber, condition, forbidden, args.max_tokens)
        model.linear.load_state_dict(trained_head)
        after_midi, after_tokens = greedy_midi(transcriber, condition, forbidden, args.max_tokens)
        result = {"index": len(pairs) + 1, "reference_notes": len(reference),
                  "before": metrics(reference, before_midi, 5.0),
                  "after": metrics(reference, after_midi, 5.0),
                  "before_tokens": before_tokens, "after_tokens": after_tokens}
        pairs.append(result)
        print("paired", result["index"], result["before"], result["after"], flush=True)
    before = mean_metrics([row["before"] for row in pairs])
    after = mean_metrics([row["after"] for row in pairs])
    result = {
        "status": "exploratory_disjoint_muscriptor_medium_5s_paired_evaluation",
        "scope": "First 5 seconds of sorted YouChorale benchmark recordings, excluding the one-song GRPO training example",
        "recording_count": len(pairs),
        "model": "MuScriptor-medium, frozen base vs 50-step token-head GRPO smoke",
        "decoding": "greedy, voice instrument constraint, max 384 tokens",
        "metric": "pitch-only 50 ms onset; complete note adds offset max(50 ms, 20% reference duration)",
        "before_macro": before, "after_macro": after,
        "paired_anonymous": pairs,
        "caveat": "One training song, one selected seed and 8 short test excerpts. Not full-song or SATB-track-aware evidence.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("macro", before, after, flush=True)


if __name__ == "__main__":
    main()
