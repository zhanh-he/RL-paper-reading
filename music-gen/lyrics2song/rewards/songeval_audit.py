"""Probe SongEval's response to controlled audio damage, without training data leakage.

Run beside the official SongEval checkout. Input audio is used for evaluation
only; this script writes aggregate scores and signal statistics, not audio.
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch


LABELS = ("Coherence", "Musicality", "Memorability", "Clarity", "Naturalness")


def signal_stats(audio):
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    at_peak = float(np.mean(np.isclose(np.abs(audio), peak, atol=1e-6)))
    return {"peak": peak, "rms": rms, "dbfs_rms": 20 * np.log10(max(rms, 1e-10)), "flat_top_fraction": at_peak}


def variants(audio, sr):
    mono = np.mean(audio, axis=0) if audio.ndim == 2 else audio
    mono = mono.astype(np.float32)
    base = mono * (0.5 / max(float(np.max(np.abs(mono))), 1e-8))
    safe = base * 1.6
    clipped = np.clip(base * 3.2, -1, 1)
    matched = clipped * (np.sqrt(np.mean(safe ** 2)) / max(np.sqrt(np.mean(clipped ** 2)), 1e-8))
    low = librosa.resample(base, orig_sr=sr, target_sr=12000)
    low = librosa.resample(low, orig_sr=12000, target_sr=sr)
    low = librosa.util.fix_length(low, size=len(base))
    return {
        "reference": base,
        "safe_gain": safe,
        "hard_clip_full_scale": clipped,
        "hard_clip_rms_matched": matched,
        "bandlimit_6khz_upsampled": low,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--songeval", type=Path, required=True)
    parser.add_argument("--audio-glob", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()
    sys.path.insert(0, str(args.songeval.resolve()))
    from eval import Synthesizer

    files = sorted(glob.glob(args.audio_glob))[:args.limit]
    if not files:
        raise ValueError("No input audio matched")
    scorer = Synthesizer(str(args.songeval / "ckpt/model.safetensors"), str(files[0]), str(args.output.parent))
    scorer.setup()
    rows = []
    for index, file in enumerate(files):
        audio, sr = librosa.load(file, sr=None, mono=False)
        for name, waveform in variants(audio, sr).items():
            reward_input = librosa.resample(waveform, orig_sr=sr, target_sr=24000)
            with torch.inference_mode():
                tensor = torch.from_numpy(reward_input).unsqueeze(0).to(scorer.device)
                hidden = scorer.muq(tensor, output_hidden_states=True)["hidden_states"][6]
                raw = scorer.model(hidden).squeeze(0).cpu().numpy()
            rows.append({"clip_index": index, "variant": name,
                         "scores": {key: float(value) for key, value in zip(LABELS, raw)},
                         "signal": signal_stats(waveform)})
        print(f"scored {index + 1}/{len(files)} clips", flush=True)
    comparisons = {}
    for variant in ("safe_gain", "hard_clip_full_scale", "hard_clip_rms_matched", "bandlimit_6khz_upsampled"):
        comparisons[variant] = {}
        for label in LABELS:
            deltas = [next(row for row in rows if row["clip_index"] == index and row["variant"] == variant)["scores"][label]
                      - next(row for row in rows if row["clip_index"] == index and row["variant"] == "reference")["scores"][label]
                      for index in range(len(files))]
            comparisons[variant][label] = {
                "mean_delta": float(np.mean(deltas)),
                "positive_count": sum(delta > 0 for delta in deltas),
                "n": len(deltas),
            }
    receipt = {"status": "controlled_perturbation_audit_not_grpo", "clips": len(files),
               "source": "local evaluation audio, not used for training or redistributed",
               "songeval_input": "librosa mono and 24kHz; MuQ hidden_states[6]",
               "comparisons": comparisons, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
