"""Strict loader allowing only byte-identical legacy duplicate VAE aliases."""
import torch


def load(cfg, dtype, device):
    from SongBloom.models.songbloom.songbloom_pl import SongBloom_PL, SongBloom_Sampler
    model = SongBloom_PL(cfg)
    state = torch.load(cfg.pretrained_path, map_location='cpu', weights_only=True)
    expected = set(model.state_dict())
    alias = 'model.condition_provider.conditioners.prompt_wav.tokenizer.'
    removed = []
    for key in list(state):
        if key in expected: continue
        canonical = 'vae.'+key[len(alias):] if key.startswith(alias) else None
        if canonical not in state or not torch.equal(state[key], state[canonical]):
            raise ValueError(f'Unverified SongBloom checkpoint key: {key}')
        removed.append(key); del state[key]
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False).eval().to(device)
    sampler = SongBloom_Sampler(model.vae, model.model.to(dtype=dtype), cfg.train_dataset.lyric_processor,
                               cfg.max_dur, cfg.sr*cfg.train_dataset.prompt_len)
    sampler.set_generation_params(**cfg.inference)
    sampler.checkpoint_compatibility = {'strict': True, 'verified_duplicate_vae_aliases': len(removed),
                                      'all_removed_tensors_exactly_equal_to_loaded_vae': True}
    print(sampler.checkpoint_compatibility, flush=True)
    return sampler
