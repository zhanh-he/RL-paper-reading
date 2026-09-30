"""Decode one public audio input with a fixed base model and GRPO head stages.

Run on the training machine. Only the MIDI and aggregate diagnostics from an
original synthetic example are suitable for the public demo; never copy the
base checkpoint or the event-head checkpoints into the repository.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--choralstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--head-run", type=Path, required=True)
    parser.add_argument("--extra-stage", action="append", default=[], metavar="STEP=HEAD_PT")
    parser.add_argument("--arm-run", action="append", default=[], metavar="ARM=RUN_DIR")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.choralstream.resolve()))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rewards"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from model import ChoralStreamModel
    from transcribe import load_mel, notes_to_midi, run_transcription
    from note_metrics import load_satb_midi, score

    config = json.loads(args.config.read_text())
    model = ChoralStreamModel(
        n_layers=config["n_layers"], seg_len=config["seg_len"],
        enable_encoder=config["enable_encoder"], prob_model=config["prob_model"],
        use_voice_queries=config["use_voice_queries"],
    ).cuda().eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.double()
    mel, _ = load_mel(input_audio=str(args.input))
    input_end = sf.info(str(args.input)).duration
    args.output.mkdir(parents=True, exist_ok=True)
    reference = load_satb_midi(args.reference)

    stages = {"0": None}
    for item in args.extra_stage:
        step, sep, path = item.partition("=")
        if not sep or not step.isdecimal() or int(step) <= 0 or step in stages:
            parser.error("--extra-stage needs a positive STEP=HEAD_PT")
        stages[step] = Path(path)
    stages.update({
        str(step): args.head_run / f"step_{step:06d}" / "event_heads.pt"
        for step in (100, 300, 1000)
    })
    for item in args.arm_run:
        arm, sep, run_dir = item.partition("=")
        if not sep or not arm.replace("_", "").isalpha():
            parser.error("--arm-run needs ARM=RUN_DIR")
        stages[f"arm_{arm}_300"] = Path(run_dir) / "step_000300" / "event_heads.pt"
    report = {"status": "public_synthetic_event_head_replay",
              "input": "original synthetic SATB audio, not held-out benchmark",
              "steps": {}}
    for step, head_path in stages.items():
        if head_path is not None:
            state = torch.load(head_path, map_location="cpu", weights_only=True)
            for key, module in (("pitch", model.trg_pitch_prj),
                                ("voice", model.trg_voice_prj),
                                ("start", model.trg_start_prj),
                                ("duration", model.trg_dur_prj)):
                module.load_state_dict(state[key], strict=True)
        with torch.inference_mode():
            prediction = run_transcription(
                model, mel, config["seg_len"], 0.5, 0.0, 0.08, 0.4, 4,
                "song", audio_path=args.input,
            )
        keep = (prediction["start"] < input_end) & (prediction["dur"] > 0)
        for key in ("pitch", "start", "dur", "voice"):
            prediction[key] = prediction[key][keep]
        prediction["dur"] = np.minimum(prediction["dur"], input_end - prediction["start"])
        active_voices = [i for i, active in enumerate(prediction["active_voice_mask"]) if active]
        stage_id = f"{int(step):04d}" if step.isdecimal() else step
        midi_path = args.output / f"choral_event_{stage_id}.mid"
        notes_to_midi(prediction["pitch"], prediction["start"],
                      prediction["dur"], prediction["voice"], midi_path,
                      active_voices=active_voices)
        metrics = score(reference, load_satb_midi(midi_path))
        report["steps"][str(step)] = {
            "midi": midi_path.name,
            "active_voices": active_voices,
            "note_count": metrics["n_estimate"],
            "frame_f1": metrics["frame"]["f1"],
            "onset_f1": metrics["onset"]["f1"],
            "onset_offset_f1": metrics["offset"]["f1"],
            "track_onset_f1": metrics["track_onset"]["f1"],
            "track_note_f1": metrics["track_note"]["f1"],
        }
        print(step, report["steps"][str(step)], flush=True)
    (args.output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
