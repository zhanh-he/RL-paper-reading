"""Read the encoded control backing tracks and attach independent accent scans.

The controls deliberately make each downbeat the only accent above 0.50 peak.
This detector validates those fixture signals. Real accompaniment needs an
independently validated downbeat tracker and an annotated vocal reference.
"""

import argparse
from array import array
import hashlib
import json
from pathlib import Path
import subprocess
import sys

RATE = 48_000
PEAK = 0.50
DEBOUNCE_S = 0.14
SEARCH_S = 0.55
TOLERANCE_MS = 80


def read_signal(path):
    pcm = subprocess.check_output([
        "ffmpeg", "-v", "error", "-i", str(path), "-ac", "1",
        "-ar", str(RATE), "-f", "f32le", "-",
    ])
    samples = array("f")
    samples.frombytes(pcm)
    if sys.byteorder != "little":
        samples.byteswap()
    return samples


def accents(samples):
    hits = []
    next_allowed = 0
    for frame, value in enumerate(samples):
        if frame >= next_allowed and abs(value) > PEAK:
            hits.append(frame / RATE)
            next_allowed = frame + round(DEBOUNCE_S * RATE)
    return hits


def inspect_bars(bars, hits):
    observations = []
    for bar in bars:
        start = bar["start_s"]
        nearby = [accent for accent in hits if abs(accent - start) <= SEARCH_S]
        observed = min(nearby, key=lambda time: abs(time - start), default=None)
        offset = round((observed - start) * 1000) if observed is not None else None
        if offset is None:
            status = "missing"
        else:
            status = "aligned" if abs(offset) <= TOLERANCE_MS else "late"
        observations.append({"index": bar["index"], "accent_s": observed, "offset_ms": offset, "status": status})
    return observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).resolve().parents[1] / "site/audio/manifest.json")
    path = parser.parse_args().manifest
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["version"] = "controls-v1"
    manifest["detector"] = {"id": "fixture-accent-v1", "peak_threshold": PEAK,
                            "search_window_ms": round(SEARCH_S * 1000), "tolerance_ms": TOLERANCE_MS}
    manifest["vocal"]["media_sha256"] = hashlib.sha256((path.parent / manifest["vocal"]["file"]).read_bytes()).hexdigest()
    for variation in manifest["variants"]:
        media = path.parent / variation["file"]
        samples = read_signal(media)
        if len(samples) != round(manifest["duration_s"] * RATE):
            raise ValueError(f"Unexpected decoded duration in {media.name}")
        hits = accents(samples)
        observations = inspect_bars(manifest["bars"], hits)
        for actual, designed in zip(observations, variation["bar_events"], strict=True):
            found, expected = actual["offset_ms"], designed["offset_ms"]
            if found is None or expected is None:
                assert found == expected, (media.name, actual, designed)
            else:
                assert abs(found - expected) <= 4, (media.name, actual, designed)
        assert len(hits) == sum(bar["accent_s"] is not None for bar in observations), hits
        variation["observed_bars"] = observations
        variation["media_sha256"] = hashlib.sha256(media.read_bytes()).hexdigest()
        print(variation["id"], [bar["offset_ms"] for bar in observations])
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
