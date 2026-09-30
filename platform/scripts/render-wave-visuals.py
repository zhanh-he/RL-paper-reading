"""Render fixed-scale waveform thumbnails for published audio examples."""

from pathlib import Path
import argparse

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw


DEMOS = Path(__file__).resolve().parents[1] / "site" / "demos"
WIDTH = 1200
HEIGHT = 140
COLOR = (26, 186, 169)


def render(audio: Path, output: Path) -> None:
    samples, _ = sf.read(audio, dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    if len(mono) == 0:
        raise ValueError(f"Empty audio: {audio}")
    edges = np.linspace(0, len(mono), WIDTH + 1, dtype=np.int64)
    image = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    center = (HEIGHT - 1) / 2
    scale = center
    for x in range(WIDTH):
        start = min(edges[x], len(mono) - 1)
        segment = mono[start:max(start + 1, edges[x + 1])]
        upper = max(-1.0, min(1.0, float(segment.max())))
        lower = max(-1.0, min(1.0, float(segment.min())))
        draw.line((x, center - upper * scale, x, center - lower * scale), fill=COLOR)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    print(output.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, nargs="+")
    args = parser.parse_args()
    for path in args.audio:
        render(path, DEMOS / "visuals" / f"{path.stem}_wave.png")
