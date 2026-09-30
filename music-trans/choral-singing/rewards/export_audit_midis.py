"""Write constructed SATB reward counterexamples as playable MIDI files."""

import argparse
from pathlib import Path

from note_metrics import VOICES, controlled_variants
from satb_reward import load_json_notes


def write_midi(path, notes):
    import mido

    midi = mido.MidiFile(ticks_per_beat=480)
    for voice in VOICES:
        track = mido.MidiTrack()
        track.append(mido.MetaMessage("track_name", name=voice, time=0))
        events = []
        for note in notes:
            if note.voice == voice:
                events.append((round(note.start * 960), 1, note.pitch))
                events.append((round(note.end * 960), 0, note.pitch))
        previous_tick = 0
        for tick, is_on, pitch in sorted(events):
            track.append(mido.Message("note_on" if is_on else "note_off", note=pitch,
                                      velocity=80 if is_on else 0, time=tick - previous_tick))
            previous_tick = tick
        midi.tracks.append(track)
    midi.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    variants = controlled_variants(load_json_notes(args.reference_json))
    for name, notes in variants.items():
        if not notes:
            continue
        write_midi(args.output_dir / f"choral_audit_{name}.mid", notes)
        print(name, len(notes))


if __name__ == "__main__":
    main()
