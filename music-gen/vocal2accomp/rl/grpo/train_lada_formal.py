"""Multi-song LaDA online GRPO, reusing the verified diffusion trajectory code."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import soundfile as sf
import torch

import train_lada_band as base
from formal_protocol import epoch_order, load_manifest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rewards"))
from combine import WEIGHTS, combine


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Worker:
    def __init__(self, cfg, out):
        self.log = (out / "reward-worker.log").open("a")
        self.process = subprocess.Popen([cfg.beat_worker_python, cfg.beat_worker_script,
                                         "--reward-root", cfg.beat_reward_root],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.log, text=True, bufsize=1)

    def score(self, row, audio):
        request = {"vocal": row["vocal"], "audio": str(audio), "seconds": row["seconds"]}
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"reward worker exited {self.process.poll()}; inspect reward-worker.log")
        result = json.loads(line)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=10)
        self.log.close()


def score_candidate(audio, row, worker, out, arm):
    path = out / "candidate.wav"
    sf.write(path, audio, 48000, subtype="FLOAT")
    beat = worker.score(row, path)
    voc, sr = sf.read(row["vocal"], dtype="float32", always_2d=True)
    voc_rms = base.frame_rms(voc, sr)
    metrics = base.score_audio(audio, voc_rms, 48000, "combined")
    rich = base.richness_v0_score(audio, voc_rms, 48000, 0.0)
    richness = float(np.clip(rich["reward"] + rich["loudness_guard_penalty"], 0, 1))
    backing, _ = sf.read(row["backing"], dtype="float32", always_2d=True)
    acc_rms = float(np.sqrt(np.mean(audio.astype("float64") ** 2)))
    gt_rms = float(np.sqrt(np.mean(backing.astype("float64") ** 2)))
    delta_db = 20 * np.log10(max(acc_rms, 1e-8) / max(gt_rms, 1e-8))
    loudness_penalty = 0.025 * max(0.0, abs(delta_db) - 6.0)
    peak = float(np.max(np.abs(audio)))
    clipping = float(np.mean(np.abs(audio) >= 0.999))
    # Check stereo channels before writing any listening-normalized derivative.
    quality = max(0.0, (peak - 0.9) * 5) + 2 * clipping + max(0.0, metrics["spectral_flatness"] - 0.35)
    result = combine(float(beat["score"]), float(beat["stft_coverage"]), richness,
                     arm=arm, quality_penalty=quality, loudness_penalty=loudness_penalty,
                     beat_scorable=bool(beat["scorable"]))
    return {**metrics, **result, "beat_confidence": beat["confidence"],
            "beat_reasons": beat["reasons"], "beat_components": beat["components"],
            "reference_events": beat["reference_beats"], "accompaniment_beats": beat["accompaniment_beats"],
            "layer_activity": rich["layer_activity"], "layer_movement": rich["layer_movement"],
            "tonality_gate": rich["tonality_gate"], "stereo_peak": peak,
            "stereo_clipped_fraction": clipping, "acc_vs_reference_rms_db": float(delta_db)}


def append(path, record):
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)


def canonical_audio(audio, seconds):
    samples = round(48000 * seconds)
    if len(audio) < samples or not np.isfinite(audio).all():
        raise ValueError("Codec output is too short or nonfinite; do not silently pad it")
    return audio[:samples]


def run(cfg):
    out = Path(cfg.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(cfg.manifest, "train")
    valid = load_manifest(cfg.validation, "valid")
    if {r["song"] for r in rows} & {r["song"] for r in valid}:
        raise ValueError("Train/validation song leakage")
    if cfg.group < 2:
        raise ValueError("GRPO requires at least two candidates")
    if (out / "run.json").exists() and not cfg.resume:
        raise ValueError("Existing run requires explicit --resume")
    sys.path.insert(0, str(Path(cfg.code_root).resolve()))
    from omegaconf import OmegaConf
    from lada_band.module.llada import ModelModule
    from lada_band.utils.config_utils import resolve_relative_paths
    config_path = Path(cfg.code_root) / "lada_band/conf/infer_1B.yaml"
    config = OmegaConf.create(resolve_relative_paths(OmegaConf.to_container(OmegaConf.load(config_path), resolve=True), str(config_path)))
    torch.manual_seed(cfg.seed)
    started = time.monotonic()
    module = ModelModule(config)
    module.load_model(cfg.checkpoint)
    module = module.cuda().eval()
    module.requires_grad_(False)
    model = module.model
    adapter = base.OutputLoRA(model.to_logits, cfg.rank, cfg.alpha).cuda()
    model.to_logits = adapter
    optimizer = torch.optim.AdamW([adapter.a.weight, adapter.b.weight], lr=cfg.lr, weight_decay=0)
    first = 0
    if cfg.resume:
        saved = torch.load(cfg.resume, map_location="cuda", weights_only=False)
        if saved["manifest_sha256"] != sha256(cfg.manifest) or saved["arm"] != cfg.arm:
            raise ValueError("Resume protocol mismatch")
        adapter.a.load_state_dict(saved["a"])
        adapter.b.load_state_dict(saved["b"])
        optimizer.load_state_dict(saved["optimizer"])
        first = saved["step"]
    total = cfg.steps or cfg.epochs * len(rows)
    metadata = {"args": vars(cfg), "train_clips": len(rows), "train_songs": len({r["song"] for r in rows}),
                "train_minutes": sum(r["seconds"] for r in rows) / 60,
                "manifest_sha256": sha256(cfg.manifest), "validation_sha256": sha256(cfg.validation),
                "planned_steps": total, "reward_version": "bcr-v1", "weights": WEIGHTS[cfg.arm],
                "optimizer_scope": "rank32 output-head LoRA; frozen backbone, codec, CLaMP3",
                "algorithm": "single-pass group-relative diffusion policy gradient with sampled KL; clipping inactive at initial ratio=1"}
    (out / "run.json").write_text(json.dumps(metadata, indent=2))
    append(out / "events.jsonl", {"event": "ready", "load_seconds": time.monotonic() - started, **metadata})
    worker = Worker(cfg, out)
    stopping = [False]
    signal.signal(signal.SIGTERM, lambda *_: stopping.__setitem__(0, True))
    signal.signal(signal.SIGUSR1, lambda *_: stopping.__setitem__(0, True))

    def save(step):
        state = {"a": adapter.a.state_dict(), "b": adapter.b.state_dict(),
                 "optimizer": optimizer.state_dict(), "step": step, "arm": cfg.arm,
                 "manifest_sha256": sha256(cfg.manifest), "args": vars(cfg)}
        temp = out / "checkpoint.tmp"
        torch.save(state, temp)
        temp.replace(out / f"step_{step:06d}.pt")

    def encode(row):
        vocal, _ = base.read_vocal(Path(row["vocal"]), row["seconds"], torch.device("cuda"))
        ids, cond = base.encode_condition(module, vocal, row["text"])
        # Codec outputs created in inference mode must be ordinary tensors for backward.
        return ids.clone(), cond.clone()

    def evaluate(step, limit):
        for index, row in enumerate(valid[:limit]):
            ids, cond = encode(row)
            for seed in cfg.eval_seeds:
                tokens, _ = base.sample_trajectory(model, ids, cond, cfg, seed)
                audio = canonical_audio(base.decode(module, tokens), row["seconds"])
                metrics = score_candidate(audio, row, worker, out, cfg.arm)
                path = out / "replays" / f"step_{step:06d}_{row['id']}_seed{seed}.wav"
                path.parent.mkdir(exist_ok=True)
                sf.write(path, audio, 48000, subtype="FLOAT")
                append(out / "evaluations.jsonl", {"event": "validation", "step": step,
                       "id": row["id"], "song": row["song"], "seed": seed,
                       "audio": str(path), "audio_sha256": sha256(path), **metrics})

    try:
        if first == 0:
            save(0)
            evaluate(0, cfg.eval_limit)
        order, current_epoch = [], -1
        for step in range(first + 1, total + 1):
            epoch, position = divmod(step - 1, len(rows))
            if epoch != current_epoch:
                order, current_epoch = epoch_order(rows, cfg.seed, epoch), epoch
            row = order[position]
            begin = time.monotonic()
            ids, cond = encode(row)
            scores, groups = [], []
            for candidate in range(cfg.group):
                tokens, actions = base.sample_trajectory(model, ids, cond, cfg, cfg.seed + step * cfg.group + candidate)
                audio = canonical_audio(base.decode(module, tokens), row["seconds"])
                scores.append(score_candidate(audio, row, worker, out, cfg.arm))
                groups.append(actions)
            rewards = np.array([s["reward"] for s in scores])
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
            stats = base.update(model, adapter, ids, cond, groups, advantages, cfg, optimizer)
            append(out / "metrics.jsonl", {"event": "train", "step": step, "epoch": epoch,
                   "id": row["id"], "song": row["song"], "seconds": row["seconds"],
                   "elapsed_seconds": time.monotonic() - begin, "scores": scores,
                   "reward_std": float(rewards.std()), "advantages": advantages.tolist(),
                   "adapter_norm": float(adapter.b.weight.norm()),
                   "vram_gb": torch.cuda.max_memory_allocated() / 2**30, **stats})
            if step in cfg.save_steps or step % cfg.save_every == 0 or step % len(rows) == 0 or step == total or stopping[0]:
                save(step)
                if not stopping[0]:
                    evaluate(step, len(valid) if step % len(rows) == 0 or step == total else cfg.eval_limit)
            if stopping[0] or time.monotonic() - started > cfg.max_seconds:
                save(step)
                append(out / "events.jsonl", {"event": "budget_checkpoint", "step": step})
                return
        (out / "train.complete").touch()
    finally:
        worker.close()


def parse_args():
    parser = argparse.ArgumentParser()
    for name in ("code-root", "checkpoint", "manifest", "validation", "output",
                 "beat-worker-python", "beat-worker-script", "beat-reward-root"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--arm", choices=list(WEIGHTS), default="bcr")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--group", type=int, default=4)
    parser.add_argument("--denoise-steps", type=int, default=16)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--support-mix", type=float, default=1e-4)
    parser.add_argument("--schedule", default="cosine")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[777, 778])
    parser.add_argument("--eval-limit", type=int, default=3)
    parser.add_argument("--save-steps", nargs="+", type=int, default=[5, 50, 100, 300, 1000])
    parser.add_argument("--save-every", type=int, default=300)
    parser.add_argument("--max-seconds", type=float, default=82800)
    parser.add_argument("--resume")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
