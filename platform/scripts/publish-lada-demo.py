"""Publish only measured LaDA-Band vocal2accomp audio and diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


STEPS = (0, 5, 50, 100, 300)
CONFIG_FIELDS = ("reward", "group", "seconds", "eval_seconds", "denoise_steps", "lr", "top_k", "top_p", "support_mix", "eval_seed", "schedule", "text")
METRIC_FIELDS = ("reward", "rms_coverage", "onset_fit_proxy", "band_occupancy_proxy", "spectral_flatness", "peak", "clipping_fraction", "quality_penalty")


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def publish(run_dir: Path, site_dir: Path, slug: str = "combined") -> Path:
    if slug not in {"combined", "coverage"}:
        raise ValueError(f"unsupported reward arm: {slug}")
    run = json.loads((run_dir / "run.json").read_text())
    evaluations = {int(item["step"]): item for item in read_jsonl(run_dir / "evaluations.jsonl")}
    audio_dir = site_dir / "audio"
    visual_dir = site_dir / "visuals"
    audio_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)
    vocal = audio_dir / "ace_emma_vocal_12s.wav"
    if not vocal.is_file():
        raise FileNotFoundError(f"fixed vocal not found: {vocal}")
    stages = []
    baseline_audio = run_dir / "step_0000.wav"
    baseline_hash = hashlib.sha256(baseline_audio.read_bytes()).digest() if baseline_audio.is_file() else None
    beat_path = run_dir / "beat_v2.json"
    beat_scores = json.loads(beat_path.read_text()) if beat_path.is_file() else {}
    for step in STEPS:
        item = evaluations.get(step)
        source = run_dir / f"step_{step:04d}.wav"
        if item is None or not source.is_file():
            continue
        name = f"lada_emma_{slug}_step_{step:04d}"
        audio = audio_dir / f"{name}.wav"
        mixture = audio_dir / f"{name}_mix.wav"
        spectrum = visual_dir / f"{name}_spectrum.png"
        shutil.copy2(source, audio)
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(vocal), "-i", str(audio),
            "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=first:normalize=0[m]",
            "-map", "[m]", "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", str(mixture),
        ], check=True)
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(audio),
            "-lavfi", "showspectrumpic=s=960x540:legend=disabled", "-frames:v", "1", str(spectrum),
        ], check=True)
        stages.append({
            "step": step,
            "audio": f"./audio/{mixture.name}",
            "accomp_audio": f"./audio/{audio.name}",
            "spectrum": f"./visuals/{spectrum.name}",
            "same_as_baseline": step > 0 and hashlib.sha256(source.read_bytes()).digest() == baseline_hash,
            "beat_v2": beat_scores.get(str(step)),
            **{key: item[key] for key in METRIC_FIELDS if key in item},
        })

    curve = []
    for item in read_jsonl(run_dir / "metrics.jsonl"):
        scores = item.get("scores", [])
        if not scores:
            continue
        curve.append({
            "step": int(item["step"]),
            "reward": sum(score["reward"] for score in scores) / len(scores),
            "rms_coverage": sum(score["rms_coverage"] for score in scores) / len(scores),
        })

    args = run["args"]
    payload = {
        "status": "measured" if stages else "pending",
        "config": {key: args[key] for key in CONFIG_FIELDS if key in args},
        "stages": stages,
        "train_curve": curve,
        "source": "ACE Studio Vocal Synth, Emma, original melody, dry mono export",
        "note": "One-source controlled replay; no paired ground-truth accompaniment or held-out song-level claim.",
    }
    output = site_dir / ("vocal-lada-run.json" if slug == "combined" else f"vocal-lada-{slug}-run.json")
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--site-dir", type=Path, required=True)
    parser.add_argument("--slug", choices=["combined", "coverage"], default="combined")
    args = parser.parse_args()
    print(publish(args.run_dir, args.site_dir, args.slug))
