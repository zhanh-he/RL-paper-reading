"""Small-data online GRPO for LaDA-Band's masked-diffusion policy head.

The remask schedule is random and exogenous, so recorded masked-token actions
have a well-defined likelihood under the replayed policy. This trains a LoRA
adapter on the output projection; the backbone, codec, and CLaMP3 stay frozen.
"""

from __future__ import annotations

import argparse
import atexit
import json
import logging
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.nn import functional as F
from torchaudio.functional import resample


@dataclass
class Action:
    state: torch.Tensor
    token: torch.Tensor
    active: torch.Tensor
    old_logp: torch.Tensor


class BeatV2Client:
    def __init__(self, python: str, script: Path, reward_root: Path, vocal: Path, log: Path):
        self.stderr = log.open("w")
        self.process = subprocess.Popen(
            [python, str(script), "--reward-root", str(reward_root), "--vocal", str(vocal)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            text=True,
            bufsize=1,
        )
        atexit.register(self.close)

    def score(self, audio: Path, seconds: float) -> dict:
        self.process.stdin.write(json.dumps({"audio": str(audio), "seconds": seconds}) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"Beat-v2 worker exited with code {self.process.poll()}")
        result = json.loads(line)
        if "error" in result:
            raise RuntimeError(f"Beat-v2 worker: {result['error']}")
        return result

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.stdin.close()
            self.process.wait(timeout=10)
        if not self.stderr.closed:
            self.stderr.close()


class OutputLoRA(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.a = nn.Linear(base.in_features, rank, bias=False)
        self.b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)
        self.scale = alpha / rank

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        base = self.base(hidden).float()
        with torch.autocast("cuda", enabled=False):
            delta = self.b(self.a(hidden.float()))
        return base + self.scale * delta


def distribution(logits: torch.Tensor, top_k: int, top_p: float, support_mix: float) -> torch.Tensor:
    from lada_band.utils.sampling import top_k_sampling, top_p_sampling

    logits = logits.float()
    full_logp = F.log_softmax(logits, dim=-1)
    if top_p < 1:
        logits = top_p_sampling(logits, top_p=top_p)
    if top_k > 0:
        logits = top_k_sampling(logits, top_k=top_k)
    filtered_logp = F.log_softmax(logits, dim=-1)
    if support_mix <= 0:
        return filtered_logp
    # Preserve the released top-k/top-p sound while retaining full support for PPO ratios.
    return torch.logaddexp(
        filtered_logp + math.log1p(-support_mix),
        full_logp + math.log(support_mix),
    )


def read_vocal(path: Path, seconds: float, device: torch.device) -> tuple[torch.Tensor, np.ndarray]:
    audio, sr = sf.read(path, frames=round(seconds * sf.info(path).samplerate), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    audio = torch.from_numpy(audio.T.copy()).to(device)
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    if sr != 48000:
        audio = resample(audio, sr, 48000)
    return audio, mono


def frame_rms(audio: np.ndarray, sr: int, hop_seconds: float = 0.04) -> np.ndarray:
    x = audio.mean(axis=1) if audio.ndim == 2 else audio
    hop = max(1, round(sr * hop_seconds))
    count = len(x) // hop
    if count == 0:
        return np.zeros(1)
    return np.sqrt(np.mean(x[: count * hop].reshape(count, hop) ** 2, axis=1) + 1e-12)


def score_audio(audio: np.ndarray, vocal_rms: np.ndarray, sr: int, arm: str) -> dict[str, float]:
    rms = frame_rms(audio, sr)
    active = rms > 0.01  # -40 dBFS frame RMS; named rms_coverage, not Beat-v2.
    coverage = float(active.mean())
    n = min(len(rms), len(vocal_rms))
    if n > 2:
        acc_onset = np.maximum(0, np.diff(np.log1p(100 * rms[:n])))
        voc_onset = np.maximum(0, np.diff(np.log1p(100 * vocal_rms[:n])))
        denom = float(np.linalg.norm(acc_onset) * np.linalg.norm(voc_onset))
        onset_fit = float(np.dot(acc_onset, voc_onset) / denom) if denom > 1e-9 else 0.0
    else:
        onset_fit = 0.0
    x = audio.mean(axis=1) if audio.ndim == 2 else audio
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    clipping = float(np.mean(np.abs(x) >= 0.98)) if len(x) else 0.0
    # A broad-band noise bed can fool coverage; spectral flatness guards it.
    band_occupancy = 0.0
    if len(x) >= 4096:
        chunks = x[: len(x) // 4096 * 4096].reshape(-1, 4096)
        full_spectrum = np.abs(np.fft.rfft(chunks, axis=1)) + 1e-8
        spectrum = full_spectrum[:, 8:700]
        flatness = float(np.mean(np.exp(np.mean(np.log(spectrum), axis=1)) / np.mean(spectrum, axis=1)))
        frequency = np.fft.rfftfreq(4096, 1 / sr)
        bands = (80, 160, 320, 640, 1280, 2560, 5120, 8000)
        power = np.mean(full_spectrum**2, axis=0)
        shares = np.asarray([power[(frequency >= left) & (frequency < right)].sum() for left, right in zip(bands[:-1], bands[1:])])
        band_occupancy = float(np.mean(shares / max(shares.sum(), 1e-9) > 0.04))
    else:
        flatness = 0.0
    quality_penalty = max(0.0, (peak - 0.9) * 5) + 2 * clipping + max(0.0, flatness - 0.35)
    reward = coverage if arm == "coverage" else 0.45 * coverage + 0.35 * onset_fit + 0.20 * band_occupancy - quality_penalty
    return {
        "reward": float(reward),
        "rms_coverage": coverage,
        "onset_fit_proxy": onset_fit,
        "band_occupancy_proxy": band_occupancy,
        "spectral_flatness": flatness,
        "peak": peak,
        "clipping_fraction": clipping,
        "quality_penalty": quality_penalty,
    }


def encode_condition(module, vocal: torch.Tensor, text: str) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.inference_mode():
        module.codec.device = vocal.device
        codec_dtype = next(module.codec_model.parameters()).dtype
        voc_ids = module.codec.sound2code(vocal.to(codec_dtype))[0, 0].unsqueeze(0)
        text_clamp = module.clamp3(texts=[text])
        condition = torch.cat([torch.zeros_like(text_clamp), text_clamp], dim=1)
    return voc_ids, condition


def sample_trajectory(model, voc_ids, condition, cfg, seed: int) -> tuple[torch.Tensor, list[Action]]:
    from lada_band.utils.remask_schedule import build_remask_ratios

    device = voc_ids.device
    voc = voc_ids.squeeze(0)
    length = voc.numel()
    prefix = condition.shape[1]
    mask_token = model.special_tokens["acc_mask"]
    state = torch.full_like(voc, mask_token)
    attention = torch.ones(1, length, dtype=torch.bool, device=device)
    ratios = build_remask_ratios(cfg.denoise_steps, cfg.schedule)
    actions = []
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        for i, ratio in enumerate(ratios):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model.forward(voc_ids, state[None], attention, condition).squeeze(0)[prefix:]
            logp = distribution(logits, cfg.top_k, cfg.top_p, cfg.support_mix)
            sampled = torch.multinomial(logp.exp(), 1).squeeze(-1)
            active = state == mask_token
            old = logp.gather(-1, sampled[:, None]).squeeze(-1)
            actions.append(Action(state.cpu(), sampled.cpu(), active.cpu(), old.cpu()))
            tokens = torch.where(active, sampled, state)
            if i == cfg.denoise_steps - 1:
                break
            num_remask = max(1, min(int(active.sum()) - 1, math.floor(ratio * length)))
            new_mask = model._select_remask_mask(
                cur_mask=active,
                sel_probs=torch.ones_like(old),
                num_to_mask=num_remask,
                strategy="random",
                seed=seed,
                sequence_id="fixed_vocal",
                step_index=i,
                selection_temperature=0.0,
            )
            state = torch.where(new_mask, mask_token, tokens)
    return tokens.detach(), actions


def decode(module, tokens: torch.Tensor) -> np.ndarray:
    with torch.inference_mode():
        sound = module.codec.code2sound(tokens.reshape(1, 1, -1))
    sound = sound.detach().float().cpu().squeeze().numpy()
    if sound.ndim == 2 and sound.shape[0] < sound.shape[1]:
        sound = sound.T
    return sound


def update(model, adapter, voc_ids, condition, groups, advantages, cfg, optimizer) -> dict[str, float]:
    device = voc_ids.device
    length = voc_ids.shape[1]
    attention = torch.ones(1, length, dtype=torch.bool, device=device)
    prefix = condition.shape[1]
    optimizer.zero_grad(set_to_none=True)
    total = sum(len(actions) for actions in groups)
    loss_sum = 0.0
    kl_sum = 0.0
    for actions, advantage in zip(groups, advantages):
        for action in actions:
            state = action.state.to(device)
            token = action.token.to(device)
            active = action.active.to(device)
            old_logp = action.old_logp.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, hidden = model.forward(voc_ids, state[None], attention, condition, return_hidden_states=True)
            current = distribution(logits.squeeze(0)[prefix:], cfg.top_k, cfg.top_p, cfg.support_mix)
            new_logp = current.gather(-1, token[:, None]).squeeze(-1)[active]
            previous = old_logp[active]
            ratio = (new_logp - previous).clamp(-20, 20).exp()
            clipped = ratio.clamp(1 - cfg.clip, 1 + cfg.clip)
            policy_loss = -torch.minimum(ratio * advantage, clipped * advantage).mean()
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    reference = adapter.base(hidden.detach())
                ref_logp = distribution(reference.squeeze(0)[prefix:], cfg.top_k, cfg.top_p, cfg.support_mix)
                ref = ref_logp.gather(-1, token[:, None]).squeeze(-1)[active]
            log_ref_ratio = (ref - new_logp).clamp(-20, 20)
            kl = (log_ref_ratio.exp() - log_ref_ratio - 1).mean()
            loss = (policy_loss + cfg.beta * kl) / total
            loss.backward()
            loss_sum += float(loss.detach())
            kl_sum += float(kl.detach()) / total
    torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
    optimizer.step()
    return {"loss": loss_sum, "sampled_kl": kl_sum}


def run(cfg):
    code_root = Path(cfg.code_root).resolve()
    sys.path.insert(0, str(code_root))
    from omegaconf import OmegaConf
    from lada_band.module.llada import ModelModule
    from lada_band.utils.config_utils import resolve_relative_paths

    out = Path(cfg.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    model_config = code_root / "lada_band/conf/infer_1B.yaml"
    config_dict = OmegaConf.to_container(OmegaConf.load(model_config), resolve=True)
    model_cfg = OmegaConf.create(resolve_relative_paths(config_dict, str(model_config)))
    start = time.monotonic()
    torch.manual_seed(cfg.seed)
    module = ModelModule(model_cfg)
    module.load_model(cfg.checkpoint)
    module = module.cuda().eval()
    module.requires_grad_(False)
    model = module.model
    adapter = OutputLoRA(model.to_logits, cfg.rank, cfg.alpha).cuda()
    model.to_logits = adapter
    optimizer = torch.optim.AdamW([adapter.a.weight, adapter.b.weight], lr=cfg.lr, weight_decay=0)
    start_step = 0
    if cfg.resume:
        previous = torch.load(cfg.resume, map_location="cuda", weights_only=False)
        adapter.a.load_state_dict(previous["a"])
        adapter.b.load_state_dict(previous["b"])
        optimizer.load_state_dict(previous["optimizer"])
        start_step = int(previous["step"])
        if start_step >= cfg.steps:
            raise ValueError(f"resume checkpoint step {start_step} is already >= target {cfg.steps}")
    vocal, mono = read_vocal(Path(cfg.vocal), cfg.seconds, torch.device("cuda"))
    vocal_rms = frame_rms(mono, sf.info(cfg.vocal).samplerate)
    voc_ids, condition = encode_condition(module, vocal, cfg.text)
    eval_vocal, eval_mono = read_vocal(Path(cfg.vocal), cfg.eval_seconds, torch.device("cuda"))
    eval_vocal_rms = frame_rms(eval_mono, sf.info(cfg.vocal).samplerate)
    eval_voc_ids, eval_condition = encode_condition(module, eval_vocal, cfg.text)
    beat_client = None
    if cfg.reward == "beat_v2":
        if not (cfg.beat_worker_python and cfg.beat_worker_script and cfg.beat_reward_root):
            raise ValueError("beat_v2 requires --beat-worker-python, --beat-worker-script and --beat-reward-root")
        beat_client = BeatV2Client(
            cfg.beat_worker_python,
            Path(cfg.beat_worker_script).resolve(),
            Path(cfg.beat_reward_root).resolve(),
            Path(cfg.vocal).resolve(),
            out / "beat_v2_worker.log",
        )

    def score_rollout(audio: np.ndarray, reference_rms: np.ndarray, seconds: float, path: Path | None = None):
        if beat_client is None:
            return score_audio(audio, reference_rms, 48000, cfg.reward)
        metrics = score_audio(audio, reference_rms, 48000, "combined")
        if path is None:
            path = out / "beat_v2_candidate.wav"
            sf.write(path, audio, 48000)
        beat = beat_client.score(path, seconds)
        metrics["proxy_reward"] = metrics["reward"]
        metrics["reward"] = float(beat["score"]) if beat["scorable"] else 0.0
        metrics["beat_v2_score"] = beat["score"]
        metrics["beat_v2_reference_beats"] = beat["reference_beats"]
        metrics["beat_v2_accompaniment_beats"] = beat["accompaniment_beats"]
        return metrics

    print(json.dumps({"event": "ready", "load_seconds": round(time.monotonic() - start, 2), "tokens": voc_ids.shape[1], "vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}), flush=True)
    evaluation_log = out / "evaluations.jsonl"
    if start_step == 0:
        evaluation_log.write_text("")

    def evaluation(step: int):
        model.eval()
        tokens, _ = sample_trajectory(model, eval_voc_ids, eval_condition, cfg, cfg.eval_seed)
        audio = decode(module, tokens)
        path = out / f"step_{step:04d}.wav"
        sf.write(path, audio, 48000)
        metrics = score_rollout(audio, eval_vocal_rms, cfg.eval_seconds, path)
        record = {"event": "evaluation", "step": step, "audio": str(path), **metrics}
        with evaluation_log.open("a") as log:
            log.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        return metrics

    if start_step == 0:
        baseline = evaluation(0)
        with (out / "run.json").open("w") as f:
            json.dump({"args": vars(cfg), "baseline": baseline, "official_model": str(cfg.checkpoint)}, f, indent=2)
    with (out / "metrics.jsonl").open("a" if start_step else "w") as log:
        for step in range(start_step + 1, cfg.steps + 1):
            began = time.monotonic()
            scores, groups = [], []
            for candidate in range(cfg.group):
                tokens, actions = sample_trajectory(model, voc_ids, condition, cfg, cfg.seed + step * cfg.group + candidate)
                audio = decode(module, tokens)
                scores.append(score_rollout(audio, vocal_rms, cfg.seconds))
                groups.append(actions)
            rewards = np.asarray([s["reward"] for s in scores])
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
            stats = update(model, adapter, voc_ids, condition, groups, advantages, cfg, optimizer)
            record = {"step": step, "elapsed_seconds": round(time.monotonic() - began, 2), "scores": scores, "advantages": advantages.tolist(), **stats}
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps({"event": "train", **record}), flush=True)
            if step in cfg.save_steps:
                torch.save({
                    "a": adapter.a.state_dict(),
                    "b": adapter.b.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "args": vars(cfg),
                }, out / f"step_{step:04d}.pt")
                evaluation(step)
    if beat_client is not None:
        beat_client.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocal", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--text", default="piano and soft drums, supportive accompaniment")
    parser.add_argument("--seconds", type=float, default=4)
    parser.add_argument("--eval-seconds", type=float, default=12)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--denoise-steps", type=int, default=8)
    parser.add_argument("--group", type=int, default=2)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--support-mix", type=float, default=1e-4)
    parser.add_argument("--schedule", default="cosine")
    parser.add_argument("--reward", choices=["coverage", "combined", "beat_v2"], default="coverage")
    parser.add_argument("--beat-worker-python")
    parser.add_argument("--beat-worker-script")
    parser.add_argument("--beat-reward-root")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--eval-seed", type=int, default=777)
    parser.add_argument("--save-steps", type=int, nargs="+", default=[5, 50, 100, 300])
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run(parse_args())
