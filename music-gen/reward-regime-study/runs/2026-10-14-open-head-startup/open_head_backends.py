"""Native generators with a narrowly trainable output head and explicit policy."""
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from open_head_policy import HeadPolicy


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def seed(value):
    import random
    random.seed(value); np.random.seed(value % (2**32)); torch.manual_seed(value); torch.cuda.manual_seed_all(value)


class DiffRhythm2:
    kind = 'gaussian-transition'

    def __init__(self, root, sources, seconds=60):
        import os
        import espeakng_loader
        os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = str(espeakng_loader.get_library_path())
        os.environ['ESPEAK_DATA_PATH'] = str(espeakng_loader.get_data_path())
        source = sources/'diffrhythm2'; sys.path.insert(0, str(source))
        self.native = load_module('diffrhythm2_native_inference', source/'inference.py')
        self.native.hf_hub_download = lambda repo_id, filename, **kwargs: str(root/filename)
        self.model, self.mulan, self.tokenizer, self.decoder = self.native.prepare_model('ASLP-lab/DiffRhythm2', 'cuda')
        self.native.lrc_tokenizer = self.tokenizer
        self.model.requires_grad_(False).eval().half(); self.decoder.requires_grad_(False).eval().half()
        self.head = HeadPolicy(self.model.transformer.proj_out)
        self.model.transformer.proj_out = self.head
        self.seconds = seconds; self.rows = []; self.noise_level = .1; self.steps = 32; self.style_cache = {}
        self.scheduler = type('PinnedSchedule', (), {})()
        self.scheduler.sigmas = torch.linspace(1, 0, self.steps, device='cuda')
        self.scheduler.timesteps = self.scheduler.sigmas[:-1]*1000
        self.scheduler.index_for_timestep = lambda t: int((self.scheduler.timesteps-t).abs().argmin())
        sys.path.insert(0, str(sources/'flow_grpo'))
        from flow_grpo.diffusers_patch.sd3_sde_with_logprob import sde_step_with_logprob
        self.sde = sde_step_with_logprob
        import diffrhythm2.cfm as cfm
        cfm.odeint = self.odeint

    def odeint(self, fn, initial, times, **kwargs):
        x = initial.float()
        for i, time in enumerate(times[:-1]):
            self.head.features = []; self.head.capture = True
            fn(time, x.half())
            self.head.capture = False
            if len(self.head.features) != 2: raise ValueError('Expected conditional and null DiT heads')
            features = torch.stack([h[0] for h in self.head.features])
            # Native CFG is pred + strength*(pred-null), so strength2 means scale3.
            velocity = self.head.guided(features.to('cuda'), 3.).unsqueeze(0)
            timestep = self.scheduler.timesteps[i:i+1]
            nxt, _, mean, std, sqrt_dt = self.sde(self.scheduler, -velocity, timestep, x,
                noise_level=self.noise_level, return_sqrt_dt=True)
            scale = std*sqrt_dt
            old = torch.distributions.Normal(mean, scale).log_prob(nxt).sum()
            self.rows.append({'features': features, 'x': x.cpu(), 'next': nxt.cpu(),
                              'timestep': timestep.cpu(), 'old_logprob': old.cpu(), 'cfg': 3.})
            x = nxt
        return torch.stack([initial.float(), x]).to(initial.dtype)

    def sample(self, row, value):
        seed(value); self.rows = []
        text = torch.tensor(sum(self.native.parse_lyrics(row['lyrics']), []), device='cuda').long()[None]
        if row['style'] not in self.style_cache:
            with torch.no_grad(): self.style_cache[row['style']] = self.mulan(texts=[row['style']]).detach().cpu()
        style = self.style_cache[row['style']].to('cuda').half()
        with torch.no_grad():
            latent = self.model.sample_block_cache(text, int(self.seconds*5), style,
                steps=self.steps, cfg_strength=2., process_bar=False)
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                seed(42); audio = self.decoder.decode_audio(latent.transpose(1, 2), overlap=5, chunk_size=20)
        return audio.float().cpu().numpy().reshape(-1), int(self.decoder.h.sampling_rate), self.rows

    def terms(self, record):
        features = record['features'].to('cuda').clone()
        x, nxt, time = (record[k].to('cuda') for k in ['x', 'next', 'timestep'])
        velocity = self.head.guided(features, 3.).unsqueeze(0)
        ref_velocity = self.head.guided(features, 3., reference=True).unsqueeze(0)
        _, _, mean, std, sqrt_dt = self.sde(self.scheduler, -velocity, time, x,
            noise_level=self.noise_level, prev_sample=nxt, return_sqrt_dt=True)
        _, _, ref_mean, _ = self.sde(self.scheduler, -ref_velocity, time, x,
            noise_level=self.noise_level, prev_sample=nxt)
        scale = std*sqrt_dt
        logprob = torch.distributions.Normal(mean, scale).log_prob(nxt).sum()
        kl_per_dimension = .5*((mean-ref_mean)/scale).square().mean()
        return logprob, kl_per_dimension

    def park(self, device):
        for module in [self.model, self.mulan, self.decoder]: module.to(device)
        if device == 'cpu': torch.cuda.empty_cache()

    def metadata(self):
        return {'policy': 'DiT proj_out-only rank8 LoRA; frozen transformer, frontends and decoder',
                'sampler': 'Full native block sequence; all31 transitions/block stochastic; official Flow-GRPO SDE',
                'steps_per_block': self.steps-1, 'noise_level': self.noise_level, 'cfg': 3.,
                'kl_unit': 'conditional Gaussian KL per latent dimension, averaged over sampled transitions',
                'logprob_unit': 'sum over the complete block latent; no omitted variance terms',
                'baseline': 'same SDE/noise schedule, zero adapter; not deterministic ODE baseline', 'fake_stereo': False}


class HeartMuLa:
    kind = 'categorical-head'

    def __init__(self, root, sources, seconds=60):
        sys.path.insert(0, str(sources/'heartlib'/'src'))
        from heartlib.pipelines.music_generation import HeartMuLaGenPipeline
        import heartlib.heartmula.modeling_heartmula as module
        HeartMuLaGenPipeline._apply_compile = lambda self, model: None
        self.pipe = HeartMuLaGenPipeline.from_pretrained(str(root), device=torch.device('cuda'),
            dtype={'mula': torch.bfloat16, 'codec': torch.float32}, version='3B', lazy_load=False)
        self.pipe.mula.requires_grad_(False).eval(); self.pipe.codec.requires_grad_(False).eval()
        self.head = HeadPolicy(self.pipe.mula.codebook0_head); self.pipe.mula.codebook0_head = self.head
        self.seconds = seconds; self.rows = []; native_sample = module.sample_topk

        def sample(logits, topk, temperature):
            if not self.head.features: return native_sample(logits, topk, temperature)
            action, record = self.head.discrete_action(1.5); self.rows.append(record)
            return action.reshape(1, 1).int()
        module.sample_topk = sample

    def sample(self, row, value):
        seed(value); self.rows = []; self.head.features = []; self.head.capture = True
        with torch.no_grad():
            inputs = self.pipe.preprocess({'tags': row['style'], 'lyrics': row['lyrics']}, cfg_scale=1.5)
            result = self.pipe._forward(inputs, max_audio_length_ms=self.seconds*1000,
                temperature=1., topk=50, cfg_scale=1.5)
            self.head.capture = False
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                seed(42); audio = self.pipe.codec.detokenize(result['frames'].to('cuda'))
        return audio.float().cpu().numpy().T, 48000, self.rows

    def terms(self, record): return self.head.discrete_terms(record)

    def park(self, device):
        self.pipe.mula.to(device); self.pipe.codec.to(device)
        if device == 'cpu': torch.cuda.empty_cache()

    def metadata(self):
        return {'policy': 'codebook0_head-only rank8 LoRA; frozen global backbone/local decoder/codec',
                'global_sampling': 'FP32 guided full-support categorical; temperature1, CFG1.5',
                'local_sampling': 'Native fixed topk50 residual-codebook environment; no trainable local parameters',
                'kl_unit': 'exact conditional categorical KL for codebook0, averaged over sampled frames',
                'limitation': 'Restricted discrete-head RL, not global/local backbone GRPO; no negative full-model conclusion'}


class SongBloom:
    kind = 'categorical-head'

    def __init__(self, root, sources, seconds=60):
        source = sources/'songbloom'; sys.path.insert(0, str(source))
        native = load_module('songbloom_native_inference', source/'infer.py')
        from SongBloom.models.songbloom.songbloom_pl import SongBloom_Sampler
        import SongBloom.models.songbloom.songbloom_mvsa as module
        cfg = native.load_config(str(root/'songbloom_full_150s.yaml'), parent_dir=str(root))
        cfg.max_dur = seconds
        self.model = SongBloom_Sampler.build_from_trainer(cfg, strict=True, dtype=torch.bfloat16, device=torch.device('cuda'))
        self.model.diffusion.requires_grad_(False).eval(); self.model.compression_model.requires_grad_(False).eval()
        self.head = HeadPolicy(self.model.diffusion.skeleton_classifier[-1])
        self.model.diffusion.skeleton_classifier[-1] = self.head
        params = dict(cfg.inference); params.update(top_k=0, top_p=0., temp=1., penalty_repeat=False)
        self.model.set_generation_params(**params); self.rows = []; native_multi = module.multinomial

        def sample(probs, num_samples=1, **kwargs):
            if not self.head.features: return native_multi(probs, num_samples=num_samples, **kwargs)
            action, record = self.head.discrete_action(float(params.get('cfg_coef', 1.5)))
            self.rows.append(record); return action.reshape(1, 1, 1).long()
        module.multinomial = sample
        self.clean = native.clean_lyrics

    def sample(self, row, value):
        seed(value); self.rows = []; self.head.features = []; self.head.capture = True
        ref = Path(row['reference_file'])
        audio, sr = sf.read(ref, dtype='float32', always_2d=True)
        import torchaudio
        prompt = torch.tensor(audio.mean(1))[None]
        if sr != 48000: prompt = torchaudio.functional.resample(prompt, sr, 48000)
        prompt = prompt[:, :480000].bfloat16()
        with torch.no_grad():
            audio = self.model.generate(self.clean(row['lyrics']), prompt)
        self.head.capture = False
        return audio[0].float().cpu().numpy().T, self.model.sample_rate, self.rows

    def terms(self, record): return self.head.discrete_terms(record)

    def park(self, device):
        self.model.diffusion.to(device); self.model.compression_model.to(device)
        if device == 'cpu': torch.cuda.empty_cache()

    def metadata(self):
        return {'policy': 'S1 skeleton_classifier final-linear-only rank8 LoRA; frozen AR backbone/flow/VAE/conditioners',
                'sampling': 'full-support guided categorical sketch tokens; native frozen acoustic environment',
                'kl_unit': 'exact conditional categorical sketch-head KL, mean over actions',
                'source': 'Community-restored150s base, not independently matched to inaccessible author endpoint',
                'limitation': 'Not acoustic Flow-GRPO, not full-backbone RL; reference audio is required'}


BACKENDS = {'diffrhythm2': DiffRhythm2, 'heartmula': HeartMuLa, 'songbloom': SongBloom}
