"""Publish only measured LaDA-Band vocal2accomp audio and diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


STEPS = (0, 5, 50, 100, 150, 200, 300)
CONFIG_FIELDS = ("reward", "group", "seconds", "eval_seconds", "denoise_steps", "lr", "top_k", "top_p", "support_mix", "eval_seed", "schedule", "text")
METRIC_FIELDS = ("reward", "rms_coverage", "onset_fit_proxy", "band_occupancy_proxy", "spectral_flatness", "peak", "clipping_fraction", "quality_penalty")


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def publish(run_dir: Path, site_dir: Path, slug: str = "combined", heldout_dir: Path | None = None) -> Path:
    if slug not in {"combined", "coverage", "beat_v2", "guarded"}:
        raise ValueError(f"unsupported reward arm: {slug}")
    run = json.loads((run_dir / "run.json").read_text())
    audio_run_dir = heldout_dir or run_dir
    if heldout_dir is None:
        evaluations = {int(item["step"]): item for item in read_jsonl(run_dir / "evaluations.jsonl")}
    else:
        evaluations = {int(item["step"]): item for item in json.loads((heldout_dir / "evaluations.json").read_text())}
    audio_dir = site_dir / "audio"
    visual_dir = site_dir / "visuals"
    audio_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)
    vocal = audio_dir / ("ace_emma_vocal_heldout_6s.wav" if heldout_dir else "ace_emma_vocal_12s.wav")
    if heldout_dir is not None:
        shutil.copy2(heldout_dir / "source.wav", vocal)
    if not vocal.is_file():
        raise FileNotFoundError(f"fixed vocal not found: {vocal}")
    vocal_spectrum = visual_dir / ("ace_emma_vocal_heldout_6s_spectrum.png" if heldout_dir else "ace_emma_vocal_12s_spectrum.png")
    visual_audio = [vocal]
    stages = []
    baseline_audio = audio_run_dir / "step_0000.wav"
    if heldout_dir is not None and not baseline_audio.is_file():
        raise FileNotFoundError(f"held-out baseline not found: {baseline_audio}")
    baseline_hash = hashlib.sha256(baseline_audio.read_bytes()).digest() if baseline_audio.is_file() else None
    beat_path = audio_run_dir / "beat_v2.json"
    beat_scores = json.loads(beat_path.read_text()) if beat_path.is_file() else {}
    for step in STEPS:
        item = evaluations.get(step)
        source = audio_run_dir / f"step_{step:04d}.wav"
        if item is None or not source.is_file():
            continue
        name = f"lada_emma_{slug}_{'heldout_' if heldout_dir else ''}step_{step:04d}"
        audio = audio_dir / f"{name}.wav"
        mixture = audio_dir / f"{name}_mix.wav"
        spectrum = visual_dir / f"{name}_spectrum.png"
        shutil.copy2(source, audio)
        visual_audio.append(audio)
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(vocal), "-i", str(audio),
            "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=first:normalize=0[m]",
            "-map", "[m]", "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", str(mixture),
        ], check=True)
        metrics = {key: item[key] for key in METRIC_FIELDS if key in item}
        if heldout_dir is not None and slug == "beat_v2":
            if str(step) not in beat_scores:
                raise ValueError(f"held-out Beat-v2 score missing for step {step}")
            metrics["reward"] = beat_scores[str(step)]["score"]
        stages.append({
            "step": step,
            "audio": f"./audio/{mixture.name}",
            "accomp_audio": f"./audio/{audio.name}",
            "spectrum": f"./visuals/{spectrum.name}",
            "same_as_baseline": step > 0 and hashlib.sha256(source.read_bytes()).digest() == baseline_hash,
            "beat_v2": beat_scores.get(str(step)),
            **metrics,
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
    config = {key: args[key] for key in CONFIG_FIELDS if key in args}
    if heldout_dir is not None and evaluations:
        config["eval_seconds"] = next(iter(evaluations.values()))["duration_seconds"]
    payload = {
        "status": "measured" if stages else "pending",
        "config": config,
        "stages": stages,
        "train_curve": curve,
        "source": "ACE Studio Vocal Synth, Emma, original melody, dry mono export",
        "source_audio": f"./audio/{vocal.name}",
        "source_spectrum": f"./visuals/{vocal_spectrum.name}",
        "phrase": "heldout" if heldout_dir else "training_excerpt",
        "note": "Same-singer held-out phrase, not a held-out song." if heldout_dir else "One-source controlled replay; no paired ground-truth accompaniment or held-out song-level claim.",
    }
    if heldout_dir is not None:
        output_name = "vocal-lada-heldout-run.json" if slug == "combined" else f"vocal-lada-{slug}-heldout-run.json"
    elif slug == "combined":
        output_name = "vocal-lada-run.json"
    else:
        output_name = f"vocal-lada-{slug}-run.json"
    output = site_dir / output_name
    subprocess.run([
        "uv", "run", "--with", "librosa", "--with", "matplotlib", "--with", "pillow",
        "python", str(Path(__file__).with_name("render-mel-visuals.py")),
        "--output-dir", str(visual_dir),
        *(str(audio) for audio in visual_audio),
    ], check=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--site-dir", type=Path, required=True)
    parser.add_argument("--slug", choices=["combined", "coverage", "beat_v2", "guarded"], default="combined")
    parser.add_argument("--heldout-dir", type=Path)
    args = parser.parse_args()
    print(publish(args.run_dir, args.site_dir, args.slug, args.heldout_dir))
