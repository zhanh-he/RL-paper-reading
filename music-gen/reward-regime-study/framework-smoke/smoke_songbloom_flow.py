"""Real native SongBloom backbone + isolated Miles/VeRL Flow-GRPO kernels.

This is NOT their distributed trainers. A receipt explicitly separates a real
model backward/update from a complete framework rollout/worker integration.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
import soundfile as sf
import torch
from backbone_lora import install, reference
from audit_flow_kernels import load, compare


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['assets', 'source', 'data', 'references', 'miles', 'verl', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--scorer', required=True)
    p.add_argument('--reward', choices=['richness','instrument','arrangement'], default='instrument'); a = p.parse_args()
    sys.path[:0] = [str(a.source), str(Path(__file__).resolve().parent.parent)]
    import torchaudio
    import SongBloom.models.songbloom.songbloom_mvsa as native
    from SongBloom.models.vae_frontend.autoencoders import AudioAutoencoder
    from load_songbloom import load as load_model
    from rewards import score_remote, sha
    DATA_SHA = 'c39d43e81793bee3f312c7028a0483b5da41068db4544acd258115043101e46f'
    spec = importlib.util.spec_from_file_location('native_infer', a.source/'infer.py')
    inference = importlib.util.module_from_spec(spec); spec.loader.exec_module(inference)
    assert sha(a.data) == DATA_SHA
    row = json.loads(a.data.read_text())['train'][0]
    refs = json.loads(a.references.read_text())
    if isinstance(refs, dict) and 'rows' in refs:
        refs = refs['rows']
    if isinstance(refs, dict):
        ref = refs[row['condition_id']]
    else:
        ref = next(r for r in refs if r['condition_id'] == row['condition_id'])
    ref_path = Path(ref if isinstance(ref, str) else ref.get('path', ref.get('local_path', ref.get('reference_file', ''))))
    assert ref_path.is_file(), ref
    a.output.mkdir(parents=True, exist_ok=True)
    cfg = inference.load_config(str(a.assets/'songbloom_full_150s.yaml'), parent_dir=str(a.assets))
    cfg.max_dur = 4
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    torch.manual_seed(42)
    sampler = load_model(cfg, dtype=dtype, device=torch.device('cuda'))
    diffusion = sampler.diffusion.eval()
    targets = install(diffusion, ['nar_dit.'])
    assert any('attn' in n for n in targets) and any('ff' in n for n in targets)
    parameters = {n:p for n,p in diffusion.named_parameters() if p.requires_grad}
    initial = {n:p.detach().cpu().clone() for n,p in parameters.items()}
    ns = load(a.miles, a.verl); backend = ns['FlowSdeStepBackend']()
    trajectory = []

    def sample(fn, initial_noise, steps, temperature, **context):
        # Native frozen planner supplies h, sketch and clean history. Only the
        # current DiT is recomputed with gradient; no cached DiT feature is reused.
        x = initial_noise.float(); schedule = torch.linspace(1, 0, 9, device='cuda')
        for i in range(8):
            sigma, sn = schedule[i:i+1], schedule[i+1:i+2]
            prediction = fn(x.to(dtype), sigma.to(dtype), **context).float()
            mu, noise, _ = backend.prev_sample_mean_and_std(prediction, x, sigma[:, None, None],
                sn[:, None, None], sigma_max=torch.full((1,1,1), .875, device='cuda'), noise_level=.1)
            nxt = mu + noise*torch.randn_like(x)
            old = backend.log_prob(nxt, mu, noise)
            trajectory.append({'x': x.cpu(), 'next': nxt.cpu(), 'sigma': sigma.cpu(), 'next_sigma': sn.cpu(),
                'old': old.cpu(), 'context': {k:v.detach().cpu() if isinstance(v,torch.Tensor) else v for k,v in context.items()}})
            x = nxt
        return x.to(dtype)
    native.sample_discrete_euler_with_temperature = sample
    original_decode = AudioAutoencoder.decode_audio

    def safe_decode(self, latents, chunked=False, overlap=32, chunk_size=128, **kwargs):
        if chunked and latents.shape[-1] < chunk_size:
            return self.decode(latents, **kwargs)
        return original_decode(self, latents, chunked=chunked, overlap=overlap, chunk_size=chunk_size, **kwargs)
    AudioAutoencoder.decode_audio = safe_decode
    sampler.set_generation_params(max_frames=100, steps=8)
    audio, sr = sf.read(ref_path, always_2d=True, dtype='float32')
    prompt = torch.tensor(audio.mean(1))[None]
    if sr != 48000:
        prompt = torchaudio.functional.resample(prompt, sr, 48000)
    prompt = prompt[:, :480000].to('cuda', dtype=dtype)
    rolls = []
    tick = time.monotonic(); torch.cuda.reset_peak_memory_stats()
    for i in range(2):
        torch.manual_seed(420000+i); trajectory.clear()
        with torch.no_grad():
            audio = sampler.generate(inference.clean_lyrics(row['lyrics']), prompt)
        assert trajectory
        folder = a.output/str(i); folder.mkdir(exist_ok=True)
        path = folder/'audio.flac'; sf.write(path, audio[0].float().cpu().numpy().T, sampler.sample_rate, subtype='PCM_24')
        rolls.append({'path': path, 'trajectory': list(trajectory)})
    diffusion.cpu(); sampler.compression_model.cpu(); torch.cuda.empty_cache()
    scores = score_remote(a.scorer, [r['path'] for r in rolls], a.reward)
    for roll, score in zip(rolls, scores):
        assert sha(roll['path']) == score['audio_sha256']
    (a.output/'rollout-scores.json').write_text(json.dumps({'reward':a.reward,'scores':scores},indent=2)+'\n')
    diffusion.cuda()
    rewards = torch.tensor([r['mean'] for r in scores], device='cuda')
    advantages = (rewards-rewards.mean())/rewards.std(unbiased=False).clamp_min(1e-8)
    if float(rewards.std(unbiased=False)) == 0:
        raise ValueError('Real reward group is tied; cannot claim a policy gradient smoke')
    # One recorded transition from each candidate checks the exact upstream
    # policy kernels with an identical backbone and trajectory on both routes.
    rows = [r['trajectory'][0] for r in rolls]
    audits = []
    for framework in ['miles', 'verl-omni']:
        with torch.no_grad():
            for name,p in parameters.items():
                p.copy_(initial[name])
        optimizer = torch.optim.AdamW(parameters.values(), lr=3e-4)
        for step in [1, 2]:
            optimizer.zero_grad(set_to_none=True)
            predictions, references = [], []
            for row_t in rows:
                context = {k:v.to('cuda') if isinstance(v,torch.Tensor) else v for k,v in row_t['context'].items()}
                x = row_t['x'].to('cuda', dtype=dtype); t = row_t['sigma'].to('cuda', dtype=dtype)
                with torch.no_grad(), reference(diffusion):
                    references.append(diffusion.diffusion_forward(x, t, **context).float())
                predictions.append(diffusion.diffusion_forward(x, t, **context).float())
            x = torch.cat([r['x'] for r in rows]).cuda(); nxt = torch.cat([r['next'] for r in rows]).cuda()
            sigmas = torch.cat([r['sigma'] for r in rows]).cuda(); sn = torch.cat([r['next_sigma'] for r in rows]).cuda()
            old = torch.cat([r['old'] for r in rows]).cuda()
            ml, vl, values = compare(ns, torch.cat(predictions), x, nxt, old, advantages, sigmas, sn, torch.cat(references), .875)
            loss = ml if framework == 'miles' else vl
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite native backbone loss')
            loss.backward()
            nonzero = [n for n,p in parameters.items() if p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0]
            assert nonzero and any('attn' in n for n in nonzero) and any('ff' in n for n in nonzero)
            norm = float(torch.nn.utils.clip_grad_norm_(parameters.values(), 1.))
            assert np.isfinite(norm)
            optimizer.step()
            delta = sum(float((p.detach().cpu()-initial[n]).float().square().sum()) for n,p in parameters.items())**.5
            assert delta > 0
            audits.append({'framework_kernel': framework, 'step': step, 'gradient_norm': norm, 'adapter_delta_l2': delta,
                'nonzero_backbone_modules': len(nonzero), **values})
    receipt = {'status': 'passed-real-SongBloom-backbone-two-updates-per-isolated-kernel', 'full_framework_e2e': False,
        'formal_training': False, 'reward': a.reward, 'condition_id': row['condition_id'], 'data_sha256': DATA_SHA, 'targets': targets,
        'device': torch.cuda.get_device_name(), 'dtype': str(dtype), 'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
        'seconds': time.monotonic()-tick, 'scores': scores, 'audits': audits,
        'limitations': ['One CMI train condition, short SDE rollouts; first transition only for kernel comparison',
                       'Distributed FSDP/Ray/vLLM-Omni/SGLang rollout engines NOT integrated or claimed passed',
                       'Miles upstream KL uses diffusion scale, not complete transition std; raw behavior preserved for audit',
                       'VeRL policy kernel plus explicit exact Gaussian reference KL; not its complete worker',
                       'SDE baseline is not the native ODE baseline; short VAE path uses unchunked decoding',
                       'No formal validation/test/MOS or audible-improvement conclusion']}
    (a.output/'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['targets','audits','scores']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
