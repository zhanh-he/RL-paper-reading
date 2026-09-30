"""Render comparable mel spectrograms for the public audio replays."""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


DEMOS = Path(__file__).resolve().parents[1] / "site" / "demos"
AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3")


def audio_for_spectrum(spectrum: Path) -> Path:
    stem = spectrum.name.removesuffix("_spectrum.png")
    for extension in AUDIO_EXTENSIONS:
        candidate = DEMOS / "audio" / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No audio for {spectrum}")


def render(audio: Path, output: Path) -> None:
    sample_rate = librosa.get_samplerate(audio)
    upper_hz = min(24_000, sample_rate / 2)
    samples, _ = librosa.load(audio, sr=None, mono=True)
    n_fft = 2048
    hop_length = 512
    spectrum = np.abs(librosa.stft(samples, n_fft=n_fft, hop_length=hop_length))
    # A full-scale sinusoid centered on an FFT bin has approximately unit amplitude.
    power = (spectrum * (4 / n_fft)) ** 2
    filters = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=256,
                                  fmin=0, fmax=upper_hz, norm=None)
    filters /= np.maximum(filters.sum(axis=1, keepdims=True), 1e-12)
    mel_power = filters @ power
    db = 10 * np.log10(np.maximum(mel_power, 1e-12))

    width, height = Image.open(output).size if output.exists() else (1200, 320)
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, facecolor="#080d13")
    ax = fig.add_axes([0.075, 0.055, 0.915, 0.925], facecolor="#080d13")
    mel_max = float(librosa.hz_to_mel(upper_hz))
    ax.imshow(db, origin="lower", aspect="auto", interpolation="nearest",
              extent=(0, len(samples) / sample_rate, 0, mel_max),
              cmap="viridis", vmin=-80, vmax=0)
    ticks_hz = [0, 1000, 4000, 8000, 12000, 16000, 20000, 24000]
    ticks_hz = [frequency for frequency in ticks_hz if frequency <= upper_hz]
    if upper_hz not in ticks_hz:
        ticks_hz = [frequency for frequency in ticks_hz if upper_hz - frequency >= 3000]
        ticks_hz.append(upper_hz)
    ax.set_yticks(librosa.hz_to_mel(ticks_hz))
    ax.set_yticklabels(["0" if hz == 0 else f"{hz / 1000:g}k" for hz in ticks_hz])
    ax.tick_params(axis="y", colors="#d5dee6", labelsize=7, length=0, pad=3)
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"{output.name}: {sample_rate} Hz audio, 0-{upper_hz:g} Hz mel")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEMOS / "visuals")
    parser.add_argument("audio", type=Path, nargs="*", help="Audio paths; omit to refresh all existing demo spectrograms")
    args = parser.parse_args()
    if args.audio:
        pairs = [(audio, args.output_dir / f"{audio.stem}_spectrum.png") for audio in args.audio]
    else:
        pairs = [(audio_for_spectrum(output), output)
                 for output in sorted(args.output_dir.glob("*_spectrum.png"))]
    for audio, output in pairs:
        render(audio, output)
