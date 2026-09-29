"""Generate an original synthetic sung-vowel guide for a public smoke demo."""

import argparse
import json
import wave
from pathlib import Path

import numpy as np


def synthesize(sample_rate=48000):
    notes = [
        (0.0, 1.3, 60), (1.45, 2.75, 64), (2.9, 4.4, 67),
        (4.75, 5.9, 64), (6.1, 7.5, 62), (7.8, 9.4, 65),
        (9.7, 11.3, 67), (11.6, 13.6, 72), (13.9, 15.3, 67),
    ]
    duration = 16.0
    audio = np.zeros(int(duration * sample_rate), dtype=np.float64)
    rng = np.random.default_rng(20260929)
    for start, end, midi in notes:
        left, right = int(start * sample_rate), int(end * sample_rate)
        local_time = np.arange(right - left) / sample_rate
        fundamental = 440.0 * 2 ** ((midi - 69) / 12)
        vibrato = 1 + 0.006 * np.sin(2 * np.pi * 5.1 * local_time)
        phase = 2 * np.pi * fundamental * np.cumsum(vibrato) / sample_rate
        tone = np.zeros_like(local_time)
        for harmonic in range(1, 18):
            frequency = harmonic * fundamental
            formants = (
                np.exp(-0.5 * ((frequency - 780) / 350) ** 2)
                + 0.5 * np.exp(-0.5 * ((frequency - 1200) / 460) ** 2)
            )
            tone += (0.15 + formants) * np.sin(harmonic * phase) / harmonic
        attack = np.minimum(1.0, local_time / 0.08)
        release = np.minimum(1.0, (end - start - local_time) / 0.16)
        envelope = np.maximum(0.0, np.minimum(attack, release))
        breath = rng.standard_normal(len(local_time)) * 0.005
        audio[left:right] += (tone + breath) * envelope
    audio /= max(np.max(np.abs(audio)), 1e-9)
    return (audio * 0.55).astype(np.float32), notes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio, notes = synthesize()
    pcm = np.clip(audio * 32767, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(48000)
        file.writeframes(pcm.tobytes())
    path.with_suffix(".json").write_text(json.dumps({
        "provenance": "original deterministic synthetic vowel guide; not human singing",
        "sample_rate": 48000, "seconds": 16.0, "notes": notes,
    }, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
