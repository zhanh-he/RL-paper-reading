#!/usr/bin/env python3
"""Generate an original guide performance and three controlled backing stems."""

import argparse
from array import array
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import wave


SYNTH_RATE = 22_050
MEDIA_RATE = 48_000
BPM = 120
BEATS_PER_BAR = 4
BAR_SECONDS = 60 / BPM * BEATS_PER_BAR
BAR_COUNT = 8
BAR_SAMPLES = round(BAR_SECONDS * SYNTH_RATE)
DURATION = BAR_COUNT * BAR_SECONDS
SAMPLE_COUNT = BAR_COUNT * BAR_SAMPLES

# The contour, rhythms and syllables below were written for this demo.
MELODY = (
    ((0.04, 0.38, 64, "da"), (0.49, 0.40, 69, "lo"), (0.98, 0.39, 72, "mi"), (1.45, 0.41, 71, "ah")),
    ((0.02, 0.63, 69, "lo"), (0.72, 0.23, 65, "da"), (1.02, 0.41, 69, "ah"), (1.50, 0.38, 72, "mi")),
    ((0.18, 0.48, 67, "ah"), (0.75, 0.21, 69, "mi"), (1.02, 0.42, 72, "lo"), (1.52, 0.38, 76, "ah")),
    ((0.03, 0.40, 74, "da"), (0.49, 0.37, 71, "lo"), (0.94, 0.74, 67, "ah")),
    ((0.02, 0.41, 72, "da"), (0.51, 0.41, 76, "mi"), (0.99, 0.43, 76, "ah"), (1.51, 0.41, 74, "lo")),
    ((0.16, 0.38, 72, "ah"), (0.66, 0.28, 69, "mi"), (1.02, 0.23, 65, "da"), (1.29, 0.62, 69, "lo")),
    ((0.09, 0.33, 67, "lo"), (0.49, 0.45, 72, "ah"), (1.03, 0.42, 76, "mi"), (1.53, 0.38, 74, "da")),
    ((0.03, 0.42, 72, "ah"), (0.54, 0.40, 69, "lo"), (1.02, 0.88, 69, "ah")),
)

# Root, then voicing. The last chord resolves to A minor.
HARMONY = (
    (45, (57, 60, 64)), (41, (53, 57, 60)), (48, (55, 60, 64)), (43, (55, 59, 62)),
    (45, (57, 60, 64)), (41, (53, 57, 60)), (48, (55, 60, 64)), (45, (57, 60, 64)),
)

FORMANTS = {
    "ah": (800, 1220, 2520),
    "da": (760, 1310, 2560),
    "lo": (454, 880, 2610),
    "mi": (320, 2040, 2870),
}


@dataclass(frozen=True)
class Style:
    id: str
    label: str
    description: str
    use_syncopation: bool = False
    inject_fault: bool = False


STYLES = (
    Style("pocket", "Straight pocket", "Quarter-note pulse, sustained keys and even cymbals."),
    Style("syncopated", "Syncopated pocket", "Offbeat chord stabs and bass pickups over a stable downbeat.", True),
    Style("drift", "Late entry + drop", "The lift enters 375 ms late, drops the next bar and locks back on bar 7.", True, True),
)


def frequency(midi: int) -> float:
    return 440 * 2 ** ((midi - 69) / 12)


def envelope(t: float, duration: float, attack: float, release: float) -> float:
    return min(1, t / attack, (duration - t) / release)


def make_vowel(midi: int, duration: float, syllable: str, noise: random.Random):
    base = frequency(midi)
    formants = FORMANTS[syllable]
    widths = (105, 190, 260)
    weights = []
    for harmonic in range(1, min(24, 5200 // int(base)) + 1):
        pitch = harmonic * base
        emphasis = sum(math.exp(-0.5 * ((pitch - center) / width) ** 2) for center, width in zip(formants, widths))
        weights.append((0.22 + emphasis) / harmonic ** 0.92)

    norm = sum(weights)
    phase = 0.0
    vibrato_phase = noise.random() * math.tau
    for sample in range(round(duration * SYNTH_RATE)):
        t = sample / SYNTH_RATE
        scoop = -32 * math.exp(-t / 0.034)
        vibrato = min(t / 0.26, 1) * 22 * math.sin(math.tau * 5.35 * t + vibrato_phase)
        phase += math.tau * base * 2 ** ((scoop + vibrato) / 1200) / SYNTH_RATE
        tone = sum(weight * math.sin((index + 1) * phase) for index, weight in enumerate(weights)) / norm
        mouth = 0.78 + 0.18 * math.sin(math.pi * t / duration)
        airy = 0.012 * (2 * noise.random() - 1)
        consonant = 0.035 * (2 * noise.random() - 1) * math.exp(-t / 0.011) if syllable == "da" else 0
        yield 0.54 * envelope(t, duration, 0.028, 0.078) * (mouth * tone + airy) + consonant


def make_kick(duration: float, volume: float):
    for sample in range(round(duration * SYNTH_RATE)):
        t = sample / SYNTH_RATE
        phase = math.tau * (49 * t + 73 * 0.045 * (1 - math.exp(-t / 0.045)))
        shape = envelope(t, duration, 0.0015, 0.016) * math.exp(-t / 0.057)
        yield volume * shape * math.sin(phase)


def make_cymbal(duration: float, volume: float, noise: random.Random, *, clap: bool = False):
    previous = 0.0
    for sample in range(round(duration * SYNTH_RATE)):
        t = sample / SYNTH_RATE
        white = 2 * noise.random() - 1
        bright_noise = white - 0.8 * previous
        previous = white
        tone = 0.22 * math.sin(math.tau * 177 * t) if clap else 0
        decay = 0.033 if clap else 0.017
        shape = envelope(t, duration, 0.004, 0.009) * math.exp(-t / decay)
        yield volume * shape * (0.68 * bright_noise + tone)


def make_tone(midi: int, duration: float, volume: float, *, bass: bool = False, soft: bool = False):
    omega = math.tau * frequency(midi)
    for sample in range(round(duration * SYNTH_RATE)):
        t = sample / SYNTH_RATE
        phase = omega * t
        if bass:
            signal = (0.74 * math.sin(phase) + 0.20 * math.sin(2 * phase) + 0.06 * math.sin(3 * phase))
            shape = envelope(t, duration, 0.016, 0.066) * math.exp(-t / 0.83)
        else:
            signal = 0.77 * math.sin(phase + 0.35 * math.exp(-t / 0.13) * math.sin(2 * phase))
            signal += 0.15 * math.sin(2.001 * phase) + 0.08 * math.sin(3.004 * phase)
            shape = envelope(t, duration, 0.031 if soft else 0.012, 0.16 if soft else 0.055)
            shape *= math.exp(-t / (1.53 if soft else 0.28))
        yield volume * shape * signal


def add_signal(track: array, start_s: float, signal, stop_sample: int) -> None:
    start = round(start_s * SYNTH_RATE)
    for index, value in enumerate(signal, start):
        if index >= stop_sample:
            return
        track[index] += value


def vocal_track() -> array:
    track = array("f", [0]) * SAMPLE_COUNT
    noise = random.Random(2191)
    for index, notes in enumerate(MELODY):
        start_s = index * BAR_SECONDS
        for offset, duration, midi, syllable in notes:
            add_signal(track, start_s + offset, make_vowel(midi, duration, syllable, noise), SAMPLE_COUNT)
    return track


def groove_track(style: Style) -> tuple[array, dict]:
    track = array("f", [0]) * SAMPLE_COUNT
    noise = random.Random(481 if style.use_syncopation else 693)
    beats = []
    downbeats = []
    offbeats = []
    bars = []
    for index, (root, chord) in enumerate(HARMONY):
        bar_start = index * BAR_SECONDS
        delay = 0.375 if style.inject_fault and index == 4 else 0
        missing = style.inject_fault and index == 5
        start_s = bar_start + delay
        stop_sample = (index + 1) * BAR_SAMPLES
        note = "Downbeat on the reference."
        if delay:
            note = "First backing downbeat comes 375 ms after the lift starts."
        elif missing:
            note = "No backing events in this bar."
        elif style.inject_fault and index == 6:
            note = "Backing returns on the reference downbeat."

        bars.append({
            "index": index + 1,
            "start_s": bar_start,
            "downbeat_s": None if missing else start_s,
            "offset_ms": None if missing else round(delay * 1000),
            "status": "missing" if missing else "late" if delay else "aligned",
            "note": note,
        })
        if missing:
            continue

        downbeats.append(start_s)
        beats.extend(start_s + beat * 0.50 for beat in range(BEATS_PER_BAR))
        add_signal(track, start_s, make_kick(0.22, 0.69), stop_sample)
        kick_offsets = (0.75, 1.25) if style.use_syncopation else (1.0,)
        for offset in kick_offsets:
            add_signal(track, start_s + offset, make_kick(0.17, 0.28), stop_sample)
        for offset in (0.5, 1.5):
            add_signal(track, start_s + offset, make_cymbal(0.12, 0.16, noise, clap=True), stop_sample)
        hat_offsets = (0.25, 0.50, 0.75, 1.12, 1.50, 1.75) if style.use_syncopation else (0.25, 0.75, 1.25, 1.75)
        for offset in hat_offsets:
            if start_s + offset < bar_start + BAR_SECONDS:
                add_signal(track, start_s + offset, make_cymbal(0.070, 0.080, noise), stop_sample)

        if style.use_syncopation:
            chord_offsets = (0.25, 0.75, 1.25, 1.75)
            chord_duration = 0.21
            bass_notes = ((0.07, root, 0.32), (0.69, root + 12, 0.19), (1.17, root + 7, 0.26), (1.66, root, 0.23))
        else:
            chord_offsets = (0.09, 1.08)
            chord_duration = 0.84
            bass_notes = ((0.08, root, 0.40), (0.56, root, 0.38), (1.08, root + 7, 0.39), (1.57, root, 0.37))
        for offset in chord_offsets:
            if start_s + offset < bar_start + BAR_SECONDS:
                if style.use_syncopation:
                    offbeats.append(start_s + offset)
                for pitch in chord:
                    add_signal(track, start_s + offset, make_tone(pitch, chord_duration, 0.086, soft=not style.use_syncopation), stop_sample)
        for offset, pitch, duration in bass_notes:
            add_signal(track, start_s + offset, make_tone(pitch, duration, 0.18, bass=True), stop_sample)

        # End every transport bar on a short fade so missing bars are truly silent.
        fade_samples = round(0.006 * SYNTH_RATE)
        for samples_left in range(1, fade_samples + 1):
            track[stop_sample - samples_left] *= (samples_left - 1) / fade_samples

    return track, {
        "id": style.id,
        "label": style.label,
        "description": style.description,
        "beats_s": beats,
        "downbeats_s": downbeats,
        "offbeats_s": offbeats,
        "bar_events": bars,
    }


def quantize(track: array) -> bytes:
    assert len(track) == SAMPLE_COUNT, "Unexpected synth duration"
    assert 0.1 < max(map(abs, track)) < 0.99, "Stem is silent or clips"
    pcm = array("h", (round(value * 32767) for value in track))
    if sys.byteorder != "little":
        pcm.byteswap()
    return pcm.tobytes()


def write_media(path: Path, track: array, use_wav: bool) -> None:
    pcm = quantize(track)
    if use_wav:
        with wave.open(str(path), "wb") as stream:
            stream.setparams((1, 2, SYNTH_RATE, 0, "NONE", "not compressed"))
            stream.writeframes(pcm)
        return
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "s16le", "-ar", str(SYNTH_RATE),
        "-ac", "1", "-i", "pipe:0", "-ar", str(MEDIA_RATE), "-c:a", "libmp3lame", "-b:a", "192k", str(path),
    ], input=pcm, check=True)


def verify_groove(track: array, metadata: dict) -> None:
    for event in metadata["bar_events"]:
        start = round(event["start_s"] * SYNTH_RATE)
        if event["status"] == "missing":
            assert not any(track[start : start + BAR_SAMPLES]), "Missing bar contains backing"
            continue
        downbeat = round(event["downbeat_s"] * SYNTH_RATE)
        assert not any(track[start:downbeat]), "Backing begins before the intended downbeat"
        assert max(map(abs, track[downbeat : downbeat + round(0.043 * SYNTH_RATE)])) > 0.55, "Downbeat accent missing"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "site" / "audio")
    parser.add_argument("--wav", action="store_true", help="Write PCM WAV instead of web-ready 192 kbps MP3.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    ext = "wav" if args.wav else "mp3"

    guide_name = f"vocal.{ext}"
    write_media(args.output / guide_name, vocal_track(), args.wav)
    variants = []
    for style in STYLES:
        audio, metadata = groove_track(style)
        verify_groove(audio, metadata)
        metadata["file"] = f"{style.id}.{ext}"
        write_media(args.output / metadata["file"], audio, args.wav)
        variants.append(metadata)

    manifest = {
        "duration_s": DURATION,
        "bpm": BPM,
        "beats_per_bar": BEATS_PER_BAR,
        "sample_rate": SYNTH_RATE if args.wav else MEDIA_RATE,
        "synthesis_sample_rate": SYNTH_RATE,
        "provenance": "Original procedural audio, melody and accompaniment written for this demo. No input recordings or model outputs.",
        "vocal": {"file": guide_name, "label": "Dry vocal guide"},
        "bars": [
            {
                "index": index + 1,
                "start_s": index * BAR_SECONDS,
                "end_s": (index + 1) * BAR_SECONDS,
                "section": "Verse" if index < 4 else "Lift",
                "vocal_onsets_s": [round(index * BAR_SECONDS + offset, 3) for offset, *_ in notes],
            }
            for index, notes in enumerate(MELODY)
        ],
        "variants": variants,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(variants) + 1} {DURATION:g}-second audio stems and manifest to {args.output}")


if __name__ == "__main__":
    main()
