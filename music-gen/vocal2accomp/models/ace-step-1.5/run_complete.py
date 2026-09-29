"""Run ACE-Step 1.5 base/complete on a fixed, authorized guide vocal."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def describe_audio(path):
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if not np.isfinite(audio).all():
        raise ValueError(f"Non-finite output: {path}")
    return {
        "path": str(Path(path).resolve()),
        "sample_rate": sample_rate,
        "seconds": len(audio) / sample_rate,
        "channels": audio.shape[1],
        "rms": float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))),
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--guide", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music

    root = Path(args.project_root).resolve()
    guide = Path(args.guide).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    guide_info = describe_audio(guide)
    handler = AceStepHandler()
    status, ok = handler.initialize_service(
        project_root=str(root), config_path="acestep-v15-base", device="cuda",
    )
    if not ok:
        raise RuntimeError(f"ACE-Step initialization failed: {status}")
    params = GenerationParams(
        task_type="complete",
        src_audio=str(guide),
        instruction="Complete the input track with drums, bass, keyboard:",
        caption="Warm, restrained piano pop with a steady bass pulse and gentle drums",
        instrumental=True,
        duration=guide_info["seconds"],
        inference_steps=args.steps,
        guidance_scale=7.0,
        thinking=False,
        use_cot_metas=False,
        use_cot_caption=False,
        use_cot_lyrics=False,
        use_cot_language=False,
        seed=args.seed,
    )
    config = GenerationConfig(batch_size=1, use_random_seed=False, audio_format="wav")
    result = generate_music(handler, None, params, config, save_dir=str(output))
    if not result.success or not result.audios:
        raise RuntimeError(f"ACE-Step generation failed: {result.error or result.status_message}")
    outputs = [describe_audio(item["path"]) for item in result.audios]
    if any(item["rms"] < 1e-5 for item in outputs):
        raise RuntimeError("Completion output is nearly silent")
    receipt = {
        "status": "inference_complete_not_grpo",
        "model": "ACE-Step 1.5 base",
        "seed": args.seed,
        "steps": args.steps,
        "task": "complete",
        "guide": guide_info,
        "outputs": outputs,
        "warning": "The complete task may return a vocal-plus-accompaniment mix, not an isolated accompaniment stem.",
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
