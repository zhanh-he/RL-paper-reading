"""Run the official DuET release on downloaded ASAP performances, retaining failures."""
import argparse
import hashlib
import json
import subprocess
import time
import traceback
from pathlib import Path

import torch
from duet import DuET


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(20261030)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    manifest = json.loads((args.data / "manifest.json").read_text())
    model = DuET.from_pretrained(args.weights, device="cuda")
    records = []
    for item in manifest["samples"]:
        if item["split"] not in args.split.split(","):
            continue
        record = {"id": item["id"], "work": item["work"], "status": "running"}
        begin = time.monotonic()
        try:
            source = args.data / item["files"]["midi_performance"]["path"]
            assert hashlib.sha256(source.read_bytes()).hexdigest() == item["files"]["midi_performance"]["sha256"]
            result = model.transcribe(source, args.output / item["id"])
            record.update(status="complete", report=result)
        except Exception as error:
            record.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        record["seconds"] = time.monotonic() - begin
        records.append(record)
        (args.output / "receipt.json").write_text(json.dumps({"model": "Wayne42/DuET", "metadata": model.metadata,
            "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "training_steps": 0,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "samples": records}, indent=2) + "\n")
        print(json.dumps({k: v for k, v in record.items() if k not in ["traceback", "report"]}), flush=True)


if __name__ == "__main__":
    main()
