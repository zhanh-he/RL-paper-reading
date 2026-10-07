"""Score a small audio batch with the official MuseCritic critique-then-score model."""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import soundfile as sf
import torch
import torchaudio


KEYS = ("Coherence", "Musicality", "Memorability", "Clarity", "Naturalness")


def load_audio_soundfile(source, sample_rate):
    waveform, original_sample_rate = sf.read(source, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(waveform.T.copy())
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if original_sample_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, original_sample_rate, sample_rate)
    return waveform.squeeze(0).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--pair", nargs=2, action="append", required=True,
                        metavar=("AUDIO", "OUTPUT_DIR"))
    args = parser.parse_args()

    example = args.repo / "infer" / "examples" / "input.jsonl"
    with example.open(encoding="utf-8") as source:
        record = json.loads(next(source))
    prompt = next(message["content"] for message in record["messages"]
                  if message["role"] == "user")
    prompt = re.sub(r"\n?<audio>\n?", "", prompt, flags=re.I).strip()

    deploy = args.repo / "muse_grpo" / "deploy" / "musecritic"
    sys.path.insert(0, str(deploy))
    import musecritic_serve
    musecritic_serve._load_audio = load_audio_soundfile
    MuseCriticServer = musecritic_serve.MuseCriticServer

    server = MuseCriticServer(args.model.resolve(), torch.device("cuda:0"),
                             default_max_tokens=args.max_new_tokens)
    server.load_model()
    for audio_name, output_name in args.pair:
        audio = Path(audio_name).resolve()
        output = Path(output_name).resolve()
        detail = server.predict(prompt, str(audio), max_new_tokens=args.max_new_tokens)
        scores = {key: float(detail["reward_scores"][key]) for key in KEYS}
        if not all(math.isfinite(value) for value in scores.values()):
            raise ValueError(f"Nonfinite MuseCritic scores for {audio}")
        output.mkdir(parents=True, exist_ok=True)
        (output / "result.json").write_text(json.dumps({audio.stem: scores}, indent=2) + "\n")
        (output / "critique.txt").write_text(detail["infer_critic"] + "\n")
        print(f"{audio.name}: MuseCritic mean={sum(scores.values()) / len(scores):.4f}", flush=True)


if __name__ == "__main__":
    main()
