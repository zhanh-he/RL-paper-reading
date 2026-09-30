"""Track-aware SATB note rewards and a small controlled-error audit.

MIDI loading needs mido. The scoring functions themselves use only the standard
library and accept Note objects, so they can be reused by a GRPO rollout.
"""

import argparse
import bisect
import json
import math
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path


VOICES = ("S", "A", "T", "B")


@dataclass(frozen=True)
class Note:
    pitch: int
    start: float
    end: float
    voice: str


def _tempo_map(midi):
    events = [(0, 500000)]
    for track in midi.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == "set_tempo":
                events.append((tick, msg.tempo))
    events.sort(key=lambda event: event[0])
    ticks = []
    seconds = []
    tempos = []
    total = 0.0
    last_tick = 0
    last_tempo = 500000
    for tick, tempo in events:
        total += (tick - last_tick) * last_tempo / (1_000_000 * midi.ticks_per_beat)
        if ticks and tick == ticks[-1]:
            tempos[-1] = tempo
        else:
            ticks.append(tick)
            seconds.append(total)
            tempos.append(tempo)
        last_tick = tick
        last_tempo = tempo

    def convert(tick):
        index = bisect.bisect_right(ticks, tick) - 1
        return seconds[index] + (tick - ticks[index]) * tempos[index] / (1_000_000 * midi.ticks_per_beat)

    return convert


def load_satb_midi(path):
    try:
        import mido
    except ImportError as exc:
        raise RuntimeError("Install mido to read MIDI files") from exc
    midi = mido.MidiFile(path)
    tick_to_second = _tempo_map(midi)
    notes = []
    for track in midi.tracks:
        name = next((msg.name.strip().upper() for msg in track if msg.type == "track_name"), "")
        voice = name[:1]
        if voice not in VOICES:
            continue
        active = defaultdict(list)
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                active[(msg.channel, msg.note)].append(tick)
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                starts = active[(msg.channel, msg.note)]
                if starts:
                    start_tick = starts.pop(0)
                    start = tick_to_second(start_tick)
                    end = tick_to_second(tick)
                    if end > start:
                        notes.append(Note(msg.note, start, end, voice))
    return sorted(notes, key=lambda n: (n.start, n.voice, n.pitch))


def _matches(ref, est, mode, onset_tol, offset_tol):
    if ref.pitch != est.pitch or abs(ref.start - est.start) > onset_tol:
        return False
    if mode in ("track_onset", "track_note") and ref.voice != est.voice:
        return False
    if mode in ("offset", "track_note"):
        tolerance = max(offset_tol, 0.2 * (ref.end - ref.start))
        return abs(ref.end - est.end) <= tolerance
    return True


def _counts(reference, estimate, mode, onset_tol=0.05, offset_tol=0.05):
    graph = []
    for est in estimate:
        graph.append([
            i for i, ref in enumerate(reference)
            if _matches(ref, est, mode, onset_tol, offset_tol)
        ])
    matched_ref = {}

    def augment(est_index, seen):
        for ref_index in graph[est_index]:
            if ref_index in seen:
                continue
            seen.add(ref_index)
            if ref_index not in matched_ref or augment(matched_ref[ref_index], seen):
                matched_ref[ref_index] = est_index
                return True
        return False

    tp = sum(augment(i, set()) for i in range(len(estimate)))
    return tp, len(estimate) - tp, len(reference) - tp


def prf(tp, fp, fn):
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def frame_prf(reference, estimate, hop=0.01, track_aware=False):
    def cells(notes):
        occupied = set()
        for note in notes:
            first = max(0, math.ceil(note.start / hop - 0.5))
            last = math.ceil(note.end / hop - 0.5)
            key = (note.voice, note.pitch) if track_aware else note.pitch
            occupied.update((index, key) for index in range(first, last))
        return occupied

    truth = cells(reference)
    predicted = cells(estimate)
    return prf(len(truth & predicted), len(predicted - truth), len(truth - predicted))


def active_voice_f1(reference, estimate, hop=0.5):
    end = max((note.end for note in reference + estimate), default=0.0)
    windows = max(1, int(end / hop) + 1)
    tp = fp = fn = 0
    for index in range(windows):
        left, right = index * hop, (index + 1) * hop
        ref_active = {n.voice for n in reference if n.start < right and n.end > left}
        est_active = {n.voice for n in estimate if n.start < right and n.end > left}
        tp += len(ref_active & est_active)
        fp += len(est_active - ref_active)
        fn += len(ref_active - est_active)
    return prf(tp, fp, fn)


def fragmentation_rate(reference, estimate):
    sustained = [n for n in reference if n.end - n.start >= 0.3]
    if not sustained:
        return 0.0
    fragmented = 0
    for ref in sustained:
        pieces = [
            est for est in estimate
            if est.voice == ref.voice and est.pitch == ref.pitch
            and min(est.end, ref.end) - max(est.start, ref.start) > 0.05
            and est.end - est.start < 0.8 * (ref.end - ref.start)
        ]
        fragmented += len(pieces) >= 2
    return fragmented / len(sustained)


def score(reference, estimate):
    if any(n.end <= n.start or n.voice not in VOICES for n in reference + estimate):
        raise ValueError("Notes need positive duration and a SATB voice")
    results = {
        mode: prf(*_counts(reference, estimate, mode))
        for mode in ("onset", "offset", "track_onset", "track_note")
    }
    results["frame"] = frame_prf(reference, estimate)
    results["track_frame"] = frame_prf(reference, estimate, track_aware=True)
    per_voice = {}
    for voice in VOICES:
        ref_part = [note for note in reference if note.voice == voice]
        est_part = [note for note in estimate if note.voice == voice]
        per_voice[voice] = {
            "frame": frame_prf(ref_part, est_part),
            "onset": prf(*_counts(ref_part, est_part, "onset")),
            "onset_offset": prf(*_counts(ref_part, est_part, "offset")),
        }
    results["per_voice"] = per_voice
    results["macro"] = {
        metric: sum(per_voice[voice][metric]["f1"] for voice in VOICES) / len(VOICES)
        for metric in ("frame", "onset", "onset_offset")
    }
    results["active_voice"] = active_voice_f1(reference, estimate)
    results["fragmentation_rate"] = fragmentation_rate(reference, estimate)
    results["n_reference"] = len(reference)
    results["n_estimate"] = len(estimate)
    return results


def controlled_variants(reference):
    output = {"oracle": list(reference)}
    output["silence"] = []
    output["drop_bass"] = [n for n in reference if n.voice != "B"]
    output["swap_soprano_alto"] = [
        replace(n, voice={"S": "A", "A": "S"}.get(n.voice, n.voice)) for n in reference
    ]
    output["shift_onset_100ms"] = [replace(n, start=n.start + 0.1, end=n.end + 0.1) for n in reference]
    split = []
    for note in reference:
        if note.end - note.start > 0.4:
            middle = (note.start + note.end) / 2
            split.extend([replace(note, end=middle - 0.02), replace(note, start=middle + 0.02)])
        else:
            split.append(note)
    output["fragment_sustained"] = split
    output["onset_only_short_notes"] = [
        replace(n, end=n.start + min(0.08, 0.25 * (n.end - n.start))) for n in reference
    ]
    output["overfill_octave"] = list(reference) + [replace(n, pitch=min(n.pitch + 12, 127)) for n in reference]
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--estimate")
    parser.add_argument("--controlled", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    reference = load_satb_midi(args.reference)
    if args.controlled:
        result = {name: score(reference, notes) for name, notes in controlled_variants(reference).items()}
    else:
        if not args.estimate:
            parser.error("--estimate is required unless --controlled is set")
        result = score(reference, load_satb_midi(args.estimate))
    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
