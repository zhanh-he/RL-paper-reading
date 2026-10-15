"""Native audio model/worker hooks for the actual Miles FSDP2 actor."""
from contextlib import nullcontext
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import ray
import soundfile as sf
import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from miles.backends.fsdp_utils.configs.train_pipeline_config import TrainPipelineConfig
from miles.backends.fsdp_utils.loss_hub.types import PreparedBatch
from miles.backends.fsdp_utils.models.parallel_plan import FSDPParallelPlan
from miles.backends.fsdp_utils.sde_step_backend import FlowSdeStepBackend
from lyric_frontend import fallback_language

TARGETS = r"transformer\.transformer_blocks\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"


def tensor_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().to(torch.float16).contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def adapter_cpu(model):
    state = {}
    for name, parameter in model.named_parameters():
        if 'lora_' not in name:
            continue
        if hasattr(parameter, 'to_local'):
            if torch.distributed.get_world_size() != 1:
                raise ValueError('Native sync currently supports a single FSDP rank only')
            tensor = parameter.to_local()
            if tensor.numel() != parameter.numel():
                raise ValueError('Incomplete single-rank adapter shard')
        else:
            tensor = parameter
        state[name.replace('.default', '')] = tensor.detach().cpu().clone()
    return state


def configure_paths(args):
    root = Path(args.assets).parent.parent
    os.chdir(root)
    os.environ.update(HF_HOME=str(root / "hf-cache"), HF_HUB_OFFLINE="1")
    sys.path.insert(0, str(args.source))
    import espeakng_loader
    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = str(espeakng_loader.get_library_path())
    os.environ["ESPEAK_DATA_PATH"] = str(espeakng_loader.get_data_path())


class DiffPolicy(torch.nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, x, sigma, history, text, style):
        device = x.device
        length, past, block = text.shape[1], history.shape[1], x.shape[1]
        size = length + past + block
        mask = torch.zeros((1, 1, size, size), device=device, dtype=torch.bool)
        mask[:, :, :length, :length] = True
        for i in range(0, past, block):
            mask[:, :, length+i:length+i+block, :length+i+block] = True
        mask[:, :, length+past:, :] = True
        positions = torch.cat([torch.arange(length, device=device), torch.arange(past+block, device=device)])[None]
        times = torch.cat([torch.full((1, length), -1., device=device), torch.ones((1, past), device=device),
                           (1-sigma)[:, None].expand(1, block)], 1).half()
        latent = self.transformer.latent_embed(torch.cat([history, x.half()], 1))
        predictions = []
        for tokens, conditioning in [(text, style), (torch.zeros_like(text), torch.zeros_like(style))]:
            embeddings = torch.cat([self.transformer.text_embed(tokens), latent], 1)
            velocity, *_ = self.transformer(x=embeddings, time=times, position_ids=positions,
                style_prompt=conditioning, attn_mask=mask, use_cache=False)
            predictions.append(velocity[:, -block:].float())
        return -(predictions[0] + 2 * (predictions[0]-predictions[1]))


class DiffConfig(TrainPipelineConfig):
    model_family = "diffrhythm2_audio"
    enable_autocast = True
    lora_target_modules = TARGETS

    def configure(self, args):
        self.args = args

    def prepare_cond_kwargs(self, cond, device):
        return cond

    def cfg_combine(self, pos, neg, scale, true_cfg_scale=None):
        return pos + scale * (pos-neg)

    def compute_noise_pred(self, *, model, latents_input, timesteps_input, pos_cond, **kwargs):
        return {"audio": model(x=latents_input["audio"], sigma=timesteps_input["audio"], **pos_cond)}


class DiffBackend:
    def __init__(self, config):
        self.config = config

    def load_component(self, component, *, checkpoint_path, master_dtype, materialize_weights):
        if not materialize_weights:
            raise ValueError("Initial native port is accepted for one training GPU only")
        args = self.config.args
        configure_paths(args)
        from diffrhythm2.backbones.dit import DiT
        from safetensors.torch import load_file
        config = json.loads((Path(args.assets) / "config.json").read_text())
        config["use_flex_attn"] = False
        transformer = DiT(**config)
        state = load_file(str(Path(args.assets) / "model.safetensors"))
        if not all(name.startswith("transformer.") for name in state):
            raise ValueError("Unexpected native CFM checkpoint keys")
        transformer.load_state_dict({name[len("transformer."):]: value for name, value in state.items()}, strict=True)
        transformer.requires_grad_(False)
        return DiffPolicy(transformer).to(master_dtype)

    def enable_gradient_checkpointing(self, model):
        from torch.utils.checkpoint import checkpoint
        for layer in model.transformer.transformer_blocks:
            original = layer.forward
            def forward(*args, _forward=original, **kwargs):
                if torch.is_grad_enabled():
                    return checkpoint(_forward, *args, use_reentrant=False, **kwargs)
                return _forward(*args, **kwargs)
            layer.forward = forward

    def fsdp_parallel_plan(self, model):
        return FSDPParallelPlan(no_split_modules=("LlamaNARDecoderLayer",))


def prepare(ctx, batch, **kwargs):
    if len(batch) != 1:
        raise ValueError("Native variable history requires microbatch1")
    pair = batch[0]
    def gpu(key):
        return pair[key].to(ctx.device)
    model = ctx.models["audio_backbone"]
    return PreparedBatch(latents={"audio": gpu("x")}, timesteps={"audio": gpu("sigma")},
        timesteps_for_model={"audio": gpu("sigma")}, model=model, component_name="audio_backbone",
        guidance_scale=2., use_cfg=True, cfg_batching=False, true_cfg_scale=None,
        pos_cond={key: gpu(key) for key in ["history", "text", "style"]}, neg_cond=None, joint_cond=None,
        advantage=torch.tensor([pair["advantage"]], device=ctx.device), extras=pair)


def loss(ctx, batch, prepared, *, new_pred, ref_pred, metrics, **kwargs):
    pair = prepared.extras
    x = prepared.latents["audio"]
    schedule = {key: pair[key].to(ctx.device) for key in ["sigma_max", "next_sigma"]}
    schedule = dict(sigmas=prepared.timesteps["audio"], next_sigmas=schedule["next_sigma"],
                    sigma_max=schedule["sigma_max"], noise_level=0.1)
    nxt = pair["next"].to(ctx.device)
    _, logp, mu, scale = ctx.sde_backend.sde_step_logprob(new_pred["audio"], x, prev_sample=nxt, **schedule)
    with torch.no_grad():
        _, _, ref_mu, _ = ctx.sde_backend.sde_step_logprob(ref_pred["audio"], x, prev_sample=nxt, **schedule)
    old = pair["old"].to(ctx.device)
    gap = (logp.detach()-old).abs().max()
    if ctx.rollout_id < 2 and gap > 0.02:
        raise ValueError(f"Sampling/training mean log-prob mismatch: {float(gap)}")
    ratio = (logp-old).exp()
    advantage = prepared.advantage
    policy = torch.maximum(-advantage*ratio, -advantage*ratio.clamp(0.8, 1.2))
    dt = (schedule["sigmas"]-schedule["next_sigmas"]).view(-1, 1, 1)
    variance = scale.square()*dt
    if not (variance > 0).all():
        raise ValueError("Non-stochastic transition in policy loss")
    kl = ((mu-ref_mu).square()/(2*variance)).mean()
    total = policy.sum() + ctx.args.diffusion_kl_beta*kl
    if not torch.isfinite(total):
        raise FloatingPointError("Nonfinite audio Flow-GRPO loss")
    for key, value in {"loss": total, "policy_loss": policy.sum(), "kl_loss": kl,
            "log_prob_mean_abs_diff": gap, "ratio_abs_minus_1": (ratio-1).abs().mean(),
            "clipfrac": ((ratio < .8) | (ratio > 1.2)).float().mean()}.items():
        metrics.emit_mean(key, total=value, count=1)
    return total


class NativeUpdater:
    def __init__(self, args, models):
        self.model = models["audio_backbone"]
        self.version = 0

    def connect_rollout_engines(self, engines, lock):
        if len(engines) != 1:
            raise ValueError("One native audio worker expected")
        self.worker = engines[0]

    def update_weights(self):
        # The initial implementation is one-rank FSDP; gather only adapter tensors.
        state = adapter_cpu(self.model)
        expected = tensor_hash(state)
        result = ray.get(self.worker.set_adapter.remote(state, self.version, expected))
        if result["sha256"] != expected:
            raise ValueError("Native adapter sync checksum mismatch")
        self.version += 1


class NativeWorker:
    def __init__(self, args):
        self.args = args
        configure_paths(args)
        spec = importlib.util.spec_from_file_location("native_diff_audio", Path(args.source) / "inference.py")
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        native.hf_hub_download = lambda repo_id, filename, **kwargs: str(Path(args.assets) / filename)
        self.model, self.mulan, tokenizer, self.decoder = native.prepare_model("ASLP-lab/DiffRhythm2", "cpu")
        original = tokenizer.tokenizer
        self.frontend_repairs = []
        def safe_g2p(text):
            try:
                return original(text)
            except Exception as error:
                if "Unknown language: other" not in str(error):
                    raise
                language = fallback_language(text)
                if language:
                    from g2p.g2p_generation import text_tokenizer
                    result = text_tokenizer.tokenize(text=text, sentence=text, language=language)
                else:
                    result = ("", [])
                self.frontend_repairs.append({"line_sha256": hashlib.sha256(text.encode()).hexdigest(),
                                              "policy": "Native Korean/English phonemizer or empty punctuation separator",
                                              "language": language})
                return result
        tokenizer.tokenizer = safe_g2p
        native.lrc_tokenizer = tokenizer
        self.native = native
        torch.manual_seed(42)
        self.policy = get_peft_model(DiffPolicy(self.model.transformer).requires_grad_(False),
            LoraConfig(r=8, lora_alpha=16, lora_dropout=0, target_modules=TARGETS, bias="none")).eval().half()
        self.decoder.requires_grad_(False).eval().half()
        self.mulan.requires_grad_(False).eval()
        self.backend = FlowSdeStepBackend()
        self.version = None
        self.tokens = {}
        self.styles = {}
        data = json.loads(Path(args.data).read_text())
        for split in ["train", "valid"]:
            for row in data[split]:
                tokens = sum(native.parse_lyrics(row["lyrics"]), [])
                if not tokens:
                    raise ValueError("Empty lyric condition")
                self.tokens[row["condition_id"]] = torch.tensor(tokens, dtype=torch.long)[None]
        self.offload()

    def offload(self):
        for model in [self.policy, self.mulan, self.decoder]:
            model.cpu()
        torch.cuda.empty_cache()

    def set_adapter(self, state, version, expected):
        self.offload()
        if tensor_hash(state) != expected:
            raise ValueError("Received adapter bytes changed")
        set_peft_model_state_dict(self.policy, state)
        actual = tensor_hash(get_peft_model_state_dict(self.policy))
        if actual != expected:
            raise ValueError("Applied adapter state mismatch")
        self.version = version
        return {"version": version, "sha256": actual}

    def info(self):
        return {"frontend_conditions": len(self.tokens), "frontend_repairs": self.frontend_repairs,
                "version": self.version, "targets": [n for n, p in self.policy.named_parameters() if p.requires_grad]}

    def generate(self, row, seed, destination, seconds, stochastic=True, block_index=0):
        if self.version is None:
            raise ValueError("Rollout before initial actor sync")
        from rewards import sha
        tick = time.monotonic()
        self.offload()
        condition = row["condition_id"]
        text = self.tokens[condition].cuda()
        if condition not in self.styles:
            self.mulan.cuda()
            with torch.inference_mode():
                self.styles[condition] = self.mulan(texts=[row["style"]]).detach().cpu().half()
            self.mulan.cpu()
            torch.cuda.empty_cache()
        style = self.styles[condition].cuda()
        self.policy.cuda()
        torch.manual_seed(seed)
        history = torch.zeros((1, 0, 64), device="cuda", dtype=torch.float16)
        pairs = []
        original = __import__("diffrhythm2.cfm", fromlist=["odeint"])
        old_odeint = original.odeint
        def odeint(fn, noise, times, **kwargs):
            nonlocal history
            x = noise.float()
            schedule = 1-times.float()
            clean = history.clone()
            block = clean.shape[1] // noise.shape[1]
            for i in range(len(times)-1):
                sigma, sn = schedule[i:i+1], schedule[i+1:i+2]
                velocity = -fn(times[i], x.half()).float()
                if stochastic and block == block_index and i in [len(times)//2-1, len(times)//2]:
                    maximum = schedule[1:2]
                    mu, std, _ = self.backend.prev_sample_mean_and_std(velocity, x, sigma[:, None, None],
                        sn[:, None, None], sigma_max=maximum[:, None, None], noise_level=.1)
                    nxt = mu + std*torch.randn_like(x)
                    pairs.append({"x": x.cpu(), "next": nxt.cpu(), "sigma": sigma.cpu(), "next_sigma": sn.cpu(),
                        "sigma_max": maximum.cpu(), "old": self.backend.log_prob(nxt, mu, std).cpu(),
                        "history": clean.cpu(), "text": text.cpu(), "style": style.cpu(), "version": self.version})
                else:
                    nxt = x + velocity * (sn-sigma)[:, None, None]
                x = nxt
            history = torch.cat([history, x.half()], 1)
            return torch.stack([noise.float(), x]).half()
        original.odeint = odeint
        try:
            with torch.inference_mode():
                latent = self.model.sample_block_cache(text, max(1, int(seconds*5)), style,
                    steps=32, cfg_strength=2., process_bar=False)
        finally:
            original.odeint = old_odeint
        self.policy.cpu()
        torch.cuda.empty_cache()
        self.decoder.cuda()
        with torch.inference_mode(), torch.random.fork_rng(devices=[0]):
            torch.manual_seed(42)
            audio = self.decoder.decode_audio(latent.transpose(1, 2), overlap=5, chunk_size=20)
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, audio.float().cpu().numpy().reshape(-1), int(self.decoder.h.sampling_rate), subtype="PCM_24")
        duration = audio.numel()/int(self.decoder.h.sampling_rate)
        self.offload()
        if stochastic and len(pairs) != 2:
            raise ValueError(f"Expected two stochastic transitions, got {len(pairs)}")
        return {"path": str(path), "sha256": sha(path), "seconds": duration, "pairs": pairs,
                "policy_version": self.version, "seed": seed, "generation_seconds": time.monotonic()-tick}
