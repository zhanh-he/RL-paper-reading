"""Paper-protocol SheetSage2 inference on an explicitly declared GTZAN subset."""
import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

import mir_eval
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import AutoModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(20261030)
    model = AutoModel.from_pretrained(str(args.assets / "SheetSage2"), trust_remote_code=True,
        local_files_only=True, base_model_path=str(args.assets / "MERT-v2-FullSong"), torch_dtype=torch.float32).eval().to("cuda")
    dtype = "bf16" if torch.cuda.is_bf16_supported() else "fp32"
    state = {"model": "m-a-p/SheetSage2", "precision": dtype, "gpu": torch.cuda.get_device_name(),
        "protocol": "SheetSage2 Appendix D.1: trim reference and predictions before 5s; beat/downbeat F1 with 70ms tolerance; recording macro mean.",
        "scope": "Ten-genre deterministic GTZAN-mini subset, not full 999/993-track paper cohort. Frozen inference, zero updates.",
        "training_steps": 0, "samples": []}
    for sample in json.loads((args.data / "manifest.json").read_text())["samples"]:
        row = {"id": sample["id"], "status": "running"}
        start = time.monotonic()
        try:
            audio = args.data / sample["audio"]
            assert hashlib.sha256(audio.read_bytes()).hexdigest() == sample["audio_sha256"]
            wave, sr = sf.read(audio, dtype="float32")
            if wave.ndim == 2:
                wave = wave.mean(axis=1)
            if sr != 24000:
                import math
                divisor = math.gcd(sr, 24000)
                wave = resample_poly(wave, 24000 // divisor, sr // divisor).astype(np.float32)
            out = args.output / sample["id"]
            result = model.transcribe(wave, sampling_rate=24000, output_dir=str(out), preset="paper", dtype=dtype)
            reference = np.loadtxt(args.data / sample["annotations"], ndmin=2)
            for metric, filename in [("beat_f1", "beat.lab"), ("downbeat_f1", "downbeat.lab")]:
                ref = reference[:, 0] if metric == "beat_f1" else reference[reference[:, 1] == 1, 0]
                pred = np.loadtxt(out / filename, ndmin=2)[:, 0] if (out / filename).stat().st_size else np.array([])
                ref, pred = ref[ref >= 5], pred[pred >= 5]
                row[metric] = float(mir_eval.beat.f_measure(ref, pred, f_measure_threshold=.07)) if len(ref) else None
            row.update(status="complete", events=len(result.get("events", [])))
        except Exception as error:
            row.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        row["seconds"] = time.monotonic() - start
        state["samples"].append(row)
        for metric in ["beat_f1", "downbeat_f1"]:
            values = [r.get(metric, 0) for r in state["samples"] if r.get(metric, 0) is not None]
            state[metric] = float(np.mean(values)) if values else None
        (args.output / "receipt.json").write_text(json.dumps(state, indent=2) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k != "traceback"}), flush=True)


if __name__ == "__main__":
    main()
