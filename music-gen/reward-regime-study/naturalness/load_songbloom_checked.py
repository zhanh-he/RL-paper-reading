"""Public-reproduction loader: strict keys after verified duplicate-VAE aliases.

Applies only to the archived full150s community-restored checkpoint. It does not
establish independent author-checkpoint identity or change the model license.
"""
import torch

def load(cfg,dtype,device):
 from SongBloom.models.songbloom.songbloom_pl import SongBloom_PL,SongBloom_Sampler
 network=SongBloom_PL(cfg)
 weights=torch.load(cfg.pretrained_path,map_location='cpu',weights_only=True)
 expected=network.state_dict();prefix='model.condition_provider.conditioners.prompt_wav.tokenizer.';aliases=[]
 for name in set(weights)-set(expected):
  if not name.startswith(prefix):raise RuntimeError('Unrecognized extra checkpoint key: '+name)
  equivalent='vae.'+name[len(prefix):]
  if equivalent not in weights or weights[name].shape!=weights[equivalent].shape or weights[name].dtype!=weights[equivalent].dtype or not torch.equal(weights[name],weights[equivalent]):raise RuntimeError('Duplicate alias is not identical: '+name)
  aliases.append(name)
 for name in aliases:weights.pop(name)
 network.load_state_dict(weights,strict=True)
 network.requires_grad_(False).eval().to(device)
 model=SongBloom_Sampler(network.vae,network.model.to(dtype=dtype),cfg.train_dataset.lyric_processor,cfg.max_dur,cfg.sr*cfg.train_dataset.prompt_len)
 model.set_generation_params(**cfg.inference)
 model.checkpoint_compatibility=dict(strict=True,verified_duplicate_vae_aliases=len(aliases),all_removed_tensors_exactly_equal_to_loaded_vae=True)
 return model
