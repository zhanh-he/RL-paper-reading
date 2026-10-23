"""Actual Miles FSDP2/Ray audio lifecycle and frozen-CMI formal training."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
from types import SimpleNamespace

import ray
import torch
import torch.distributed as dist
from miles.backends.fsdp_utils.actor import FSDPTrainRayActor
from miles.backends.fsdp_utils.arguments import FSDPArgs
from miles.backends.fsdp_utils import checkpoint
from miles.utils.ray_utils import Box
from diff_audio import NativeWorker, tensor_hash, adapter_cpu
from rewards import score_remote, sha

DATA_SHA = "c39d43e81793bee3f312c7028a0483b5da41068db4544acd258115043101e46f"


def write(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


class AudioActor(FSDPTrainRayActor):
    def initialize(self, args, worker):
        self.args = args
        self.last_metrics = {}
        result = self.init(args, "actor")
        self.set_rollout_manager(worker)
        self.update_weights()
        return {"start": result, "global_step": self.global_step,
                "targets": [name for name, parameter in self.model.named_parameters() if parameter.requires_grad],
                "torch": torch.__version__}

    def _log_metrics(self, rollout_id, log_dict, step):
        self.last_metrics = dict(log_dict, step=step, lr=float(self.optimizer.param_groups[0]["lr"]))
        super()._log_metrics(rollout_id, log_dict, step)

    def train_group(self, rollout_id, pairs):
        torch.cuda.reset_peak_memory_stats()
        self.train(rollout_id, [Box(ray.put({"train_data": pairs}))])
        gradients = {}
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                if not torch.isfinite(parameter.to_local() if hasattr(parameter, "to_local") else parameter).all():
                    raise FloatingPointError("Nonfinite adapter")
                if parameter.grad is not None:
                    gradient = parameter.grad.to_local() if hasattr(parameter.grad, "to_local") else parameter.grad
                    gradients[name] = float(gradient.float().norm())
        if not gradients or not all(__import__("math").isfinite(v) for v in gradients.values()):
            raise FloatingPointError("Missing or nonfinite backbone gradients")
        if not any(v > 0 and ".self_attn." in n for n, v in gradients.items()):
            raise FloatingPointError("No attention gradient")
        if not any(v > 0 and ".mlp." in n for n, v in gradients.items()):
            raise FloatingPointError("No MLP gradient")
        result = dict(self.last_metrics, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            nonzero_attention=sum(v > 0 and ".self_attn." in n for n, v in gradients.items()),
            nonzero_mlp=sum(v > 0 and ".mlp." in n for n, v in gradients.items()))
        if self.global_step <= 2 and self.global_step == 1 and abs(result["train/kl_loss"]) > 1e-8:
            raise ValueError("Reference identity failed")
        self.sleep()
        self.update_weights()
        return result

    def save(self, rollout_id):
        self.wake_up()
        self.save_model(rollout_id)
        result = self.inspect()
        self.sleep()
        return result

    def inspect(self):
        state = adapter_cpu(self.model)
        optimizer = self.optimizer.state_dict()
        moments = sum(float(value.detach().double().sum()) for values in optimizer["state"].values()
                      for key, value in values.items() if key in ["exp_avg", "exp_avg_sq"])
        return {"global_step": self.global_step, "adapter_sha256_fp16": tensor_hash(state),
                "optimizer_moment_sum": moments, "lr": self.optimizer.param_groups[0]["lr"],
                "torch_rng_sha256": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()}

    def close(self):
        if dist.is_initialized():
            dist.destroy_process_group()


class ManagedWorker(NativeWorker):
    def attach(self, handle):
        self.handle = handle
        self.new_engines = 1

    def get_rollout_engines_and_lock(self):
        return [self.handle], None, self.new_engines

    def clear_num_new_engines(self):
        self.new_engines = 0

    def set_train_parallel_config(self, config):
        self.parallel_config = config


def make_args(cli, output, steps, resume=False):
    values = asdict(FSDPArgs())
    values.update(
        source=str(cli.source), assets=str(cli.assets), data=str(cli.data),
        distributed_backend="nccl", distributed_timeout_minutes=10, num_gpus_per_node=1,
        debug_rollout_only=False, use_miles_dashboard=False, use_wandb=False, wandb_run_id=None,
        start_rollout_id=None, seed=42, diffusion_forward_dtype="fp16", fsdp_master_dtype="fp32",
        fsdp_reduce_dtype="fp32", train_pipeline_config_path="diff_audio.DiffConfig",
        model_backend_path="diff_audio.DiffBackend", hf_checkpoint=str(cli.assets),
        lora_adapter_paths=None, ref_load=None, teacher_load=None, update_weight_target_modules=["audio_backbone"],
        ref_lora_adapter_paths=None, teacher_lora_adapter_paths=None, ref_cpu_offload=False, teacher_cpu_offload=False,
        use_lora=True, lora_target_modules=None, lora_rank=8, lora_alpha=16, lora_init_weights="kaiming-uniform",
        optimizer="adam", lr=cli.lr, weight_decay=0., gradient_checkpointing=True,
        load=str(output/"checkpoints") if resume else None, ckpt_step=None, save=str(output/"checkpoints"),
        no_load_optim=False, no_load_rng=False, no_save_optim=False,
        offload_train=True, use_ema=False, train_only=False, rollout_weights="actor",
        native_weight_updater_path="diff_audio.NativeUpdater", num_rollout=steps,
        rollout_batch_size=1, n_samples_per_prompt=2, global_batch_size=2,
        num_steps_per_rollout=1, micro_batch_size=1, micro_batch_size_sample=None,
        diffusion_recompute_old_log_prob=False, log_loss_sigma_bucket=0, loss_type="custom_loss",
        diffusion_kl_beta=.05, ref_mode="lora_base", custom_prepare_train_batch_path="diff_audio.prepare",
        custom_loss_function_path="diff_audio.loss", sde_step_backend_path="miles.backends.fsdp_utils.sde_step_backend.FlowSdeStepBackend",
        debug_skip_optimizer_step=False, clip_grad=1., skip_train_actor_gc_collect=False,
    )
    return SimpleNamespace(**values)


def actor(worker, args):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    handle = ray.remote(num_gpus=.5, num_cpus=2)(AudioActor).remote(1, 0, "127.0.0.1", port)
    ray.get(worker.attach.remote(worker))
    info = ray.get(handle.initialize.remote(args, worker))
    if len(info["targets"]) != 224:
        raise ValueError(f"Expected112 attention/MLP LoRA modules, got {len(info['targets'])} tensors")
    return handle, info


def run_group(cli, worker, trainer, data, step, output, seconds, reward):
    row = data["train"][step % 270]
    folder = output/"train_rollouts"/f"{step+1:06d}"
    block_count = (int(seconds*5)+9)//10
    block_index = step % block_count
    candidates = [ray.get(worker.generate.remote(row, 700000+2*step+i, str(folder/f"{i}.flac"),
                   seconds, True, block_index)) for i in range(2)]
    scores = score_remote(cli.scorer, [Path(row["path"]) for row in candidates], reward)
    for sample, score in zip(candidates, scores):
        if sample["sha256"] != score["audio_sha256"]:
            raise ValueError("Score/audio SHA binding changed")
    rewards = torch.tensor([score["mean"] for score in scores])
    std = rewards.std(unbiased=False)
    advantages = (rewards-rewards.mean())/std.clamp_min(1e-8)
    pairs = []
    for sample, advantage in zip(candidates, advantages):
        for pair in sample.pop("pairs"):
            pairs.append(dict(pair, advantage=float(advantage)))
    versions = {pair["version"] for pair in pairs}
    if len(versions) != 1:
        raise ValueError("Mixed rollout policies")
    tick = time.monotonic()
    metrics = ray.get(trainer.train_group.remote(step, pairs))
    record = {"step": step+1, "reward": float(rewards.mean()), "rewards": rewards.tolist(),
        "kl": metrics["train/kl_loss"], "kl_unit": "exact conditional Gaussian KL / active latent dimension",
        "lr": metrics["lr"], "beta": .05, "condition_id": row["condition_id"],
        "zero_reward_std": bool(std == 0), "samples": candidates, "scores": scores,
        "policy_version": versions.pop(), "metrics": metrics, "update_seconds": time.monotonic()-tick}
    write(folder/"receipt.json", record)
    with (output/"steps.jsonl").open("a") as target:
        target.write(json.dumps(record, allow_nan=False)+"\n")
    write(output/"progress.json", {"state": "training", "step": step+1, "target": cli.steps, "latest": record})
    print(f"FORMAL={output.name} STEP={step+1} reward={record['reward']:.6f} KL={record['kl']:.6f} LR={record['lr']}", flush=True)
    return record


def validate(cli, worker, data, step, output, reward):
    folder = output/f"valid_{step:06d}"
    samples = []
    for index, row in enumerate(data["valid"]):
        sample = ray.get(worker.generate.remote(row, 5101+index, str(folder/f"{index:03d}.flac"),
                                               cli.seconds, False, 0))
        sample.pop("pairs")
        samples.append(sample)
        write(output/"progress.json", {"state": "validation", "step": step, "completed_conditions": index+1, "target_conditions": 30})
    scores = score_remote(cli.scorer, [Path(row["path"]) for row in samples], reward)
    for sample, score in zip(samples, scores):
        if sample["sha256"] != score["audio_sha256"]:
            raise ValueError("Validation score/audio mismatch")
    write(folder/"receipt.json", {"step": step, "n": 30, "scope": "All frozen CMI30 validation, not test",
        "mean": sum(score["mean"] for score in scores)/30, "samples": samples, "rows": scores})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["assets", "source", "data", "output"]:
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--scorer", required=True)
    parser.add_argument("--seconds", type=float, default=60.)
    parser.add_argument("--steps", type=int, default=270)
    parser.add_argument("--lr", type=float, choices=[3e-4,1e-4], default=3e-4)
    parser.add_argument("--acceptance-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rewards", nargs='+', choices=['richness','instrument','arrangement','mean7'],
                        default=['richness','instrument','arrangement'])
    cli = parser.parse_args()
    for name in ["assets", "source", "data", "output"]:
        setattr(cli, name, getattr(cli, name).resolve())
    if sha(cli.data) != DATA_SHA:
        raise ValueError("Frozen CMI projection changed")
    data = json.loads(cli.data.read_text())
    if len(data["train"]) != 270 or len(data["valid"]) != 30:
        raise ValueError("Full frozen split required")
    cli.output.mkdir(parents=True, exist_ok=True)
    ray_root = cli.output/"ray"/str(os.getpid())
    ray_root.mkdir(parents=True)
    short = Path(f"/tmp/musicrl-{os.getpid()}")
    short.symlink_to(ray_root, target_is_directory=True)
    ray.init(num_gpus=1, num_cpus=6, include_dashboard=False, _temp_dir=str(short),
             object_store_memory=1024**3, log_to_driver=True)
    worker = ray.remote(num_gpus=.5, num_cpus=2)(ManagedWorker).remote(cli)
    ray.get(worker.attach.remote(worker))
    write(cli.output/"frontend.json", ray.get(worker.info.remote()))
    acceptance = cli.output/"acceptance"
    acceptance.mkdir(exist_ok=True)
    trainer, info = actor(worker, make_args(cli, acceptance, 2))
    if 'mean7' in cli.rewards and cli.rewards != ['mean7']:
        raise ValueError('Pure Mean7 uses its own scorer and separate experiment root')
    acceptance_reward = 'mean7' if cli.rewards == ['mean7'] else 'instrument'
    first = run_group(cli, worker, trainer, data, 0, acceptance, 4., acceptance_reward)
    second = run_group(cli, worker, trainer, data, 1, acceptance, 4., acceptance_reward)
    if first["zero_reward_std"] or second["zero_reward_std"]:
        raise ValueError("Acceptance requires real non-tied reward groups")
    if second['kl'] <= 0:
        raise ValueError('Acceptance must detect drift against the frozen reference')
    before = ray.get(trainer.save.remote(1))
    ray.get(trainer.close.remote())
    ray.kill(trainer)
    trainer, restored = actor(worker, make_args(cli, acceptance, 2, resume=True))
    after = ray.get(trainer.inspect.remote())
    if before != after or restored["start"] != 2:
        raise ValueError(f"Cold actor checkpoint restore failed: {before} != {after}")
    write(acceptance/"receipt.json", {"full_framework_e2e": True, "formal_training": False,
        "scope": "Actual Miles FSDP2 actor + native Ray audio worker; two updates, checksum sync and cold-actor optimizer/RNG/cursor restore",
        "before": before, "after": after, "first": first, "second": second, "actor": info})
    ray.get(trainer.close.remote())
    ray.kill(trainer)
    if cli.acceptance_only:
        ray.shutdown()
        return
    for reward in cli.rewards:
        lr_tag = '3e4' if cli.lr == 3e-4 else '1e4'
        output = cli.output/f"diffrhythm2-{reward}-lr{lr_tag}-beta5e2-seed42"
        if (output/"completion.json").exists() and cli.resume:
            continue
        resuming = output.exists() and cli.resume
        if output.exists() and not resuming:
            raise FileExistsError("Refusing to overwrite a formal arm")
        output.mkdir(exist_ok=resuming)
        manifest = {"model": "DiffRhythm2", "framework": "Miles-Diffusion FSDP2 + project native Ray audio worker",
            "lr": cli.lr, "scheduler": "constant", "beta": .05, "group_size": 2, "seed": 42,
            "steps": cli.steps, "reward": reward, "data_sha256": DATA_SHA, "train_n": 270, "valid_n": 30,
            "seconds_budget": cli.seconds, "sampling": "native32 Euler grid; one block with2 stochastic steps; matched window within group; eta0.1, CFG2",
            "kl_unit": "exact conditional Gaussian KL / active latent dimension", "test_run": False,
            "replay_seconds_target": 60, "public_audio": "MP3 only"}
        if resuming:
            if json.loads((output/'experiment.json').read_text()) != manifest:
                raise ValueError('Resume configuration changed')
        else:
            write(output/"experiment.json", manifest)
        trainer, info = actor(worker, make_args(cli, output, cli.steps, resume=resuming))
        start = info['start']
        if not resuming:
            validate(cli, worker, data, 0, output, reward)
        elif (output/'steps.jsonl').exists():
            records = [json.loads(line) for line in (output/'steps.jsonl').open() if line.strip()]
            if len(records) < start:
                raise ValueError('Checkpoint ahead of reward journal')
            if len(records) > start:
                write(output/f'uncommitted-after-{start}.json', records[start:])
                (output/'steps.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in records[:start]))
        for step in range(start, cli.steps):
            run_group(cli, worker, trainer, data, step, output, cli.seconds, reward)
            ray.get(trainer.save.remote(step))
            if step+1 in [50, 100, cli.steps]:
                validate(cli, worker, data, step+1, output, reward)
        write(output/"completion.json", {"steps": cli.steps, "full_valid_n": 30, "test_run": False})
        ray.get(trainer.close.remote())
        ray.kill(trainer)
    ray.shutdown()


if __name__ == "__main__":
    main()
