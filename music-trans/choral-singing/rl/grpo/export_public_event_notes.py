"""Export SATB note geometry for the original public checkpoint replay."""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--midi-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extra-step", type=int, action="append", default=[])
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rewards"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from note_metrics import load_satb_midi

    paths = {"reference": args.reference, **{
        str(step): args.midi_dir / f"choral_event_{step:04d}.mid"
        for step in (0, 100, 300, 1000, *args.extra_step)
    }}
    paths.update({path.stem.removeprefix("choral_event_"): path
                  for path in args.midi_dir.glob("choral_event_arm_*.mid")})
    result = {}
    for stage, path in paths.items():
        result[stage] = [
            {"voice": "SATB".index(note.voice), "pitch": note.pitch,
             "start": round(note.start, 5), "end": round(note.end, 5)}
            for note in load_satb_midi(path)
        ]
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
