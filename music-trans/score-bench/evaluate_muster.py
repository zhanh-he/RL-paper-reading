"""Run the released MUSTER wrapper on complete ASAP reference/predicted scores."""
import argparse
import json
import time
from pathlib import Path
from muster import muster


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.data / "manifest.json").read_text())
    records = []
    for sample in manifest["samples"]:
        if sample["split"] != "test": continue
        pred = args.predictions / sample["id"] / "score.musicxml"
        if not pred.exists():
            records.append({"id": sample["id"], "status": "missing"}); continue
        begin = time.monotonic()
        result = muster(pred, args.data / sample["files"]["xml_score"]["path"], clean_scores=False)
        records.append({"id": sample["id"], "work": sample["work"], "status": "complete" if result.get("MeanER") is not None else "failed", "metrics": result, "seconds": time.monotonic()-begin})
        state = {"scorer": "TimFelixBeyer/amtevaluation.github.io MUSTER release wrapper", "protocol": "Full MusicXML score pairs; no test-time time or octave correction; clean_scores=False (direct file paths). Paper scorer, local subset, not paper-table reproduction.", "samples": records}
        complete = [r["metrics"] for r in records if r["status"] == "complete"]
        state["n_complete"] = len(complete)
        state["n_expected"] = 8
        state["mean"] = {key: sum(float(r[key]) for r in complete) / len(complete) for key in complete[0] if all(r.get(key) is not None for r in complete)} if complete else None
        (args.predictions / "muster.json").write_text(json.dumps(state, indent=2) + "\n")
        print(json.dumps(records[-1]), flush=True)


if __name__ == "__main__": main()
