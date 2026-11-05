"""Isolated external DiffRhythm2 two-conditional contrast inference (no RL).

Batch: {"cases":[{"id":..., "family":..., "arm":..., "seed":37,
 "lyrics":..., "good_style":..., "bad_style":..., "beta":0.5,
 "sham":false}]}. Velocity = v_good + beta*(v_good-v_bad).
Both branches receive identical lyric embeddings and generated history; their
style embeddings differ. A sham reuses exactly the good embedding as bad.
Only this model instance's sampler is patched. Shared source/assets stay intact.
"""
import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import textwrap
import time
import types


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha(tensor):
    import torch
    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def patch_sampler_source(source):
    """Fail closed on source drift; callable without Torch for source validation."""
    source = textwrap.dedent(source)
    replacements = [
        ('cfg_text_emb = self.transformer.text_embed(torch.zeros_like(text))',
         'cfg_text_emb = text_emb.clone()\n    self._contrast_record_prefix(text_emb, cfg_text_emb)', 1),
        ('style_prompt=torch.zeros_like(style_prompt)',
         'style_prompt=self._contrast_bad_style_for_call()', 3),
        ('return pred + (pred - null_pred) * cfg_strength',
         'self._contrast_record_velocity(pred, null_pred)\n            return pred + (pred - null_pred) * cfg_strength', 1),
    ]
    counts = {}
    for old, new, required in replacements:
        count = source.count(old)
        if count != required:
            raise ValueError(f'Sampler source drift: {old!r}: expected {required}, found {count}')
        counts[old] = count
        source = source.replace(old, new)
    if 'torch.zeros_like(style_prompt)' in source or 'self.transformer.text_embed(torch.zeros_like(text))' in source:
        raise ValueError('Unexpected remaining unconditional branch')
    compile(source, '<isolated-diffrhythm2-contrast>', 'exec')
    return source, counts


def install_contrast_sampler(model):
    import torch
    original = inspect.unwrap(model.sample_block_cache.__func__)
    source = textwrap.dedent(inspect.getsource(original))
    patched, counts = patch_sampler_source(source)
    namespace = dict(original.__globals__)
    exec(compile(patched, '<isolated-diffrhythm2-contrast>', 'exec'), namespace)
    model.sample_block_cache = types.MethodType(namespace[original.__name__], model)

    def bad_style(self):
        self._contrast_negative_style_calls += 1
        return self._contrast_bad_style

    def record_prefix(self, good, bad):
        equal = bool(torch.equal(good, bad))
        self._contrast_prefix = dict(equal=equal, good_sha256=tensor_sha(good), bad_sha256=tensor_sha(bad), shape=list(good.shape))
        if not equal:
            raise ValueError('Contrast branches received different lyric embeddings')

    def record_velocity(self, good, bad):
        self._contrast_velocity_calls += 1
        if self._contrast_first_velocity is None:
            delta = good.float() - bad.float()
            finite = bool(torch.isfinite(good).all() and torch.isfinite(bad).all())
            self._contrast_first_velocity = dict(good_sha256=tensor_sha(good), bad_sha256=tensor_sha(bad),
                delta_rms=float(delta.square().mean().sqrt()), delta_abs_max=float(delta.abs().max()), finite=finite)
            if not finite:
                raise ValueError('Nonfinite first branch velocity')

    model._contrast_bad_style_for_call = types.MethodType(bad_style, model)
    model._contrast_record_prefix = types.MethodType(record_prefix, model)
    model._contrast_record_velocity = types.MethodType(record_velocity, model)
    return dict(original_sampler_sha256=hashlib.sha256(source.encode()).hexdigest(),
                patched_sampler_sha256=hashlib.sha256(patched.encode()).hexdigest(), checked_replacements=counts,
                scope='Instance-only source copy; original module/class/shared files are unchanged')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['source', 'assets', 'runtime-root', 'batch', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--seconds', type=int, default=60)
    p.add_argument('--steps', type=int, default=32)
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    for name in ['source', 'assets', 'runtime_root', 'batch', 'output']:
        setattr(a, name, getattr(a, name).resolve())
    if a.seconds < 4 or a.seconds % 2 or a.steps < 2:
        raise ValueError('Use even native seconds >=4 and steps >=2')
    batch = json.loads(a.batch.read_text())
    cases = batch['cases']
    # Accept the registered native-wrapper keys as well as explicit branch names.
    for case in cases:
        case.setdefault('good_style', case.get('style'))
        case.setdefault('bad_style', case.get('negative_style'))
        case.setdefault('beta', case.get('cfg_strength'))
        case.setdefault('sham', case.get('arm') == 'sham10')
    if not cases or len({r['id'] for r in cases}) != len(cases):
        raise ValueError('Need nonempty unique cases')
    for r in cases:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', r['id']):
            raise ValueError('Unsafe case ID')
        if not r.get('lyrics') or not r.get('good_style') or not r.get('bad_style'):
            raise ValueError('Explicit original lyrics and both style descriptions required')
        beta = float(r['beta'])
        if not math.isfinite(beta) or beta < 0 or (0 < beta < 1e-5):
            raise ValueError('Beta must be zero or >=1e-5, finite, and nonnegative')
        if r.get('sham', False) and r['bad_style'] != r['good_style']:
            raise ValueError('Sham must explicitly record identical good and bad descriptions')
        if (a.output / (r['id'] + '.wav')).exists():
            raise FileExistsError('Refusing overwrite ' + r['id'])
    if (a.output / 'run.json').exists():
        raise FileExistsError('Refusing overwrite run receipt')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(a.source))
    os.chdir(a.runtime_root)  # Author MuQ loader resolves existing caches at cwd/ckpt.
    import torch
    import numpy as np
    import soundfile as sf
    import espeakng_loader
    torch.set_num_threads(a.threads)
    os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = str(espeakng_loader.get_library_path())
    os.environ['ESPEAK_DATA_PATH'] = str(espeakng_loader.get_data_path())
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this recorded native protocol')
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rec = json.loads((a.assets / 'asset-receipt.json').read_text())
    expected = {r['file']: r['sha256'] for r in rec['files']}
    required = ['model.safetensors', 'config.json', 'decoder.bin', 'decoder.json']
    actual = {name: file_sha(a.assets / name) for name in required}
    if any(actual[name] != expected[name] for name in required):
        raise ValueError('Author asset hash mismatch')
    source_hashes = {name: file_sha(a.source / name) for name in ['inference.py', 'diffrhythm2/cfm.py']}
    spec = importlib.util.spec_from_file_location('diff_native_inference', a.source / 'inference.py')
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)

    def local_download(repo_id, filename, **kwargs):
        if repo_id != 'ASLP-lab/DiffRhythm2' or filename not in required:
            raise ValueError('Unexpected asset request')
        return str(a.assets / filename)

    native.hf_hub_download = local_download
    a.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    model, mulan, tokenizer, decoder = native.prepare_model('ASLP-lab/DiffRhythm2', 'cuda')
    native.lrc_tokenizer = tokenizer
    model.requires_grad_(False).eval().half()
    mulan.requires_grad_(False).eval()
    decoder.requires_grad_(False).eval().half()
    if model.odeint_kwargs.get('method') != 'euler':
        raise ValueError('This registered run requires native Euler so branch call counts are exact')
    patch_receipt = install_contrast_sampler(model)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    text_cache, style_cache, conditioning, rows, noise_by_seed = {}, {}, {}, [], {}
    native_randn = torch.randn
    profile = dict(model='DiffRhythm2 author release', assets=actual, asset_revision=rec['files'][0]['revision'],
        native_source_sha256=source_hashes, sampler_patch=patch_receipt,
        device=torch.cuda.get_device_name(), torch=torch.__version__, requested_seconds=a.seconds, steps=a.steps,
        odeint_kwargs=dict(model.odeint_kwargs), load_seconds=load_seconds, batch_sha256=file_sha(a.batch),
        inference='Native cached block Euler ODE; FP16 model+decoder; instance-only two-conditional style branch; no adapters or waveform normalization',
        formula='velocity = v_good + beta * (v_good - v_bad)',
        branch_semantics='Identical lyric prefix tensor values and generated latent history; separate style-conditioned KV caches; bad branch is not unconditional',
        beta_zero='Pure good conditional; no negative ODE evaluation. Negative prefix/history caches are still built by native structure.',
        seed_protocol='External Python/NumPy/Torch/CUDA reset after conditioning; native seed argument unused; all initial block noise hashes compared on common prefix; decoder seed42 in fork_rng',
        scope='External public model feasibility experiment only; no company codec, no trained reward, no RL improvement claim',
        reproducibility='Within-run same-seed noise equality required; sham PCM/latent equality checked; no cross-hardware bitwise guarantee')

    def write_receipt(status):
        (a.output / 'run.json').write_text(json.dumps(dict(status=status, profile=profile, samples=rows), ensure_ascii=False, indent=2, allow_nan=False) + '\n')

    write_receipt('loaded')
    for case in cases:
        case_start = time.perf_counter()
        good_style, bad_style, lyrics = case['good_style'], case['bad_style'], case['lyrics']
        beta = float(case['beta'])
        sham = bool(case.get('sham', False))
        if lyrics not in text_cache:
            text_cache[lyrics] = torch.tensor(sum(native.parse_lyrics(lyrics), []), device='cuda', dtype=torch.long)[None]
        with torch.inference_mode():
            for style in {good_style, bad_style}:
                if style not in style_cache:
                    embedding = mulan(texts=[style]).detach()
                    style_cache[style] = embedding.half()
                    conditioning[style] = embedding.float().cpu().reshape(-1).tolist()
            (a.output / 'conditioning.json').write_text(json.dumps(dict(model='MuQ-MuLan-large', embeddings=conditioning), ensure_ascii=False, allow_nan=False) + '\n')
        good_emb = style_cache[good_style]
        bad_emb = good_emb if sham else style_cache[bad_style]
        if not sham and torch.equal(good_emb, bad_emb):
            raise ValueError('Different style branches collapsed to identical FP16 embeddings')
        model._contrast_bad_style = bad_emb
        model._contrast_negative_style_calls = 0
        model._contrast_velocity_calls = 0
        model._contrast_prefix = None
        model._contrast_first_velocity = None
        seed = int(case['seed'])
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        noise = []

        def tracked_randn(*args, **kwargs):
            value = native_randn(*args, **kwargs)
            if tuple(value.shape) == (1, model.block_size, model.num_channels):
                noise.append(tensor_sha(value))
            return value

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        sample_start = time.perf_counter()
        try:
            torch.randn = tracked_randn
            with torch.inference_mode():
                latent = model.sample_block_cache(text_cache[lyrics], a.seconds * 5, good_emb,
                    steps=a.steps, cfg_strength=beta, process_bar=False)
        finally:
            torch.randn = native_randn
        torch.cuda.synchronize()
        ode_seconds = time.perf_counter() - sample_start
        max_blocks = (a.seconds * 5 + model.block_size - 1) // model.block_size
        if not 1 <= len(noise) <= max_blocks or latent.shape[1] <= 0:
            raise ValueError('Invalid native early-stop or missing initial noise')
        prior = noise_by_seed.get(seed)
        if prior is not None and prior[:min(len(prior), len(noise))] != noise[:min(len(prior), len(noise))]:
            raise ValueError('Same-seed initial noise prefix mismatch; paired claim invalid')
        if prior is None or len(noise) > len(prior):
            noise_by_seed[seed] = noise
        expected_velocity_calls = len(noise) * (a.steps - 1) if beta >= 1e-5 else 0
        expected_negative_calls = 1 + len(noise) + expected_velocity_calls
        if model._contrast_negative_style_calls != expected_negative_calls or model._contrast_velocity_calls != expected_velocity_calls:
            raise ValueError('Unexpected negative branch call counts')
        if not model._contrast_prefix or not model._contrast_prefix['equal']:
            raise ValueError('Missing identical lyric-prefix receipt')
        first_velocity = model._contrast_first_velocity
        if first_velocity is not None and sham and first_velocity['delta_abs_max'] != 0:
            raise ValueError('Sham first velocities differ')
        if first_velocity is not None and not sham and first_velocity['delta_abs_max'] == 0:
            raise ValueError('Real contrast first velocities did not differ')
        decode_start = time.perf_counter()
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            with torch.inference_mode():
                wave = decoder.decode_audio(latent.transpose(1, 2), overlap=5, chunk_size=20)
        torch.cuda.synchronize()
        decode_seconds = time.perf_counter() - decode_start
        array = np.ascontiguousarray(wave.float().cpu().numpy().reshape(-1), dtype=np.float32)
        sr = int(decoder.h.sampling_rate)
        if sr != 48000 or not np.isfinite(array).all() or not 0 < len(array) / sr <= a.seconds + .001:
            raise ValueError('Invalid native output')
        path = a.output / (case['id'] + '.wav')
        sf.write(path, array, sr, subtype='FLOAT')
        embedding_cosine = float(torch.nn.functional.cosine_similarity(good_emb.float().flatten(), bad_emb.float().flatten(), dim=0))
        record = dict(case, source_kind='actual_generation', path=str(path), audio_sha256=file_sha(path),
            audio_pcm_sha256=hashlib.sha256(array.tobytes()).hexdigest(), sample_rate=sr, channels=1,
            duration_s=len(array) / sr, requested_duration_s=a.seconds, native_early_stop=len(noise) < max_blocks or latent.shape[1] < a.seconds * 5,
            latent_frames=int(latent.shape[1]), latent_sha256=tensor_sha(latent), initial_noise_sha256=noise,
            same_seed_noise_common_prefix_matches=True, complete_initial_noise_sequence=len(noise) == max_blocks,
            lyric_prefix=model._contrast_prefix, good_style_embedding_sha256=tensor_sha(good_emb), bad_style_embedding_sha256=tensor_sha(bad_emb),
            good_bad_style_cosine=embedding_cosine, negative_style_calls=model._contrast_negative_style_calls,
            velocity_pair_calls=model._contrast_velocity_calls, first_velocity=first_velocity,
            actual_sampler_kwargs=dict(duration=a.seconds * 5, steps=a.steps, cfg_strength=beta, process_bar=False, seed=None),
            ode_seconds=ode_seconds, decode_seconds=decode_seconds, decoder_seed=42,
            end_to_end_seconds=time.perf_counter() - case_start, gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            rms_dbfs=float(20 * np.log10(max(np.sqrt(np.mean(array.astype(float) ** 2)), 1e-12))),
            peak=float(np.max(np.abs(array))), human_rating=None)
        rows.append(record)
        write_receipt('running')
        print(json.dumps({k: record[k] for k in ['id', 'duration_s', 'audio_pcm_sha256', 'end_to_end_seconds', 'gpu_peak_allocated_bytes']}), flush=True)
    controls = []
    for row in rows:
        if row.get('sham', False):
            baseline = [r for r in rows if not r.get('sham', False) and float(r['beta']) == 0 and r['seed'] == row['seed'] and r['family'] == row['family'] and r['good_style'] == row['good_style'] and r['lyrics'] == row['lyrics']]
            if len(baseline) != 1:
                raise ValueError('Sham requires exactly one matched beta-zero baseline')
            base = baseline[0]
            controls.append(dict(sham_id=row['id'], baseline_id=base['id'],
                same_latent=base['latent_sha256'] == row['latent_sha256'],
                same_pcm=base['audio_pcm_sha256'] == row['audio_pcm_sha256'],
                same_noise=base['initial_noise_sha256'] == row['initial_noise_sha256']))
    profile['sham_controls'] = controls
    profile['shared_source_unchanged'] = all(file_sha(a.source / name) == sha for name, sha in source_hashes.items())
    if not profile['shared_source_unchanged']:
        write_receipt('failed_source_changed')
        raise RuntimeError('Shared source changed during run')
    failed_controls = any(not all(c[k] for k in ['same_latent', 'same_pcm', 'same_noise']) for c in controls)
    write_receipt('complete_with_failed_control' if failed_controls else 'complete')
    if failed_controls:
        raise RuntimeError('Sham no-op control failed; inspect saved receipt before interpreting contrasts')


if __name__ == '__main__':
    main()
