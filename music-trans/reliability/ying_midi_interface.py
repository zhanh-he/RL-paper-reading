"""Execute only the pinned upstream MIDI parser, not the singer model."""
import argparse
import ast
import hashlib
import json
import platform
import tempfile
from datetime import datetime, timezone
from os import PathLike
from pathlib import Path
from typing import Union

import numpy as np
import pretty_midi

COMMIT = "6d7c453781cdaf286227ba7a51e79d57b19fad71"
SOURCE_SHA = "7e2eb9aed7dc4b2adc5e51a4c07fbd7c3eee88f9bf0d8ea516c3ec6396652cbe"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.read_bytes()
    assert hashlib.sha256(source).hexdigest() == SOURCE_SHA, "Upstream source changed"
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "get_notemidi_from_file")
    # Avoid importing PyTorch or initializing any upstream neural components.
    namespace = dict(np=np, pretty_midi=pretty_midi, Union=Union, PathLike=PathLike,
                     DEFAULT_MELODY_FRAME_HZ=100)
    exec(compile(ast.Module(body=[function], type_ignores=[]), "pinned-upstream-parser", "exec"), namespace)
    rows = []
    sequences = []
    with tempfile.TemporaryDirectory(prefix="ying-midi-interface-") as directory:
        for name, pitches in [("lead-only", [72]), ("lead-then-accompaniment", [72, 60]),
                              ("accompaniment-then-lead", [60, 72])]:
            midi = pretty_midi.PrettyMIDI(initial_tempo=120)
            for pitch in pitches:
                instrument = pretty_midi.Instrument(program=0, name=f"pitch-{pitch}")
                instrument.notes.append(pretty_midi.Note(velocity=80, pitch=pitch, start=0, end=1))
                midi.instruments.append(instrument)
            path = Path(directory) / f"{name}.mid"
            midi.write(str(path))
            sequence = namespace["get_notemidi_from_file"](path)
            sequences.append(sequence)
            rows.append(dict(id=name, input_pitches=pitches, frames=len(sequence),
                             unique_output_pitches=np.unique(sequence).tolist(),
                             retained_lead_fraction=float(np.mean(sequence == 72)),
                             midi_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    assert rows[0]["retained_lead_fraction"] == 1
    assert rows[1]["retained_lead_fraction"] == 0
    assert rows[2]["retained_lead_fraction"] == 1
    changed = float(np.mean(sequences[1] != sequences[2]))
    assert changed == 1
    result = dict(
        scope="CPU execution of the upstream MIDI parser only; no model weights or singing generation",
        executed_at=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
        numpy=np.__version__, pretty_midi=pretty_midi.__version__,
        upstream_commit=COMMIT, upstream_source_sha256=SOURCE_SHA,
        upstream_url=f"https://github.com/GiantAILab/YingMusic-Singer/blob/{COMMIT}/src/singer/model.py#L188-L242",
        parser_ast_sha256=hashlib.sha256(ast.dump(function).encode()).hexdigest(),
        frame_hz=100, independent_musical_works=0, rows=rows,
        reordered_identical_note_content_changed_frame_fraction=changed,
        conclusion="On this overlapping two-track control, later notes overwrite earlier notes. Select a monophonic lead explicitly; do not feed combined score MIDI.",
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
