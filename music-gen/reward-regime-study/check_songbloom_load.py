"""CPU-only strict checkpoint/source compatibility check, not an experiment."""
import argparse
from pathlib import Path
import sys
import torch

p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); args = p.parse_args()
sys.path.insert(0, str(args.root/'sources/songbloom'))
import infer
from load_songbloom import load
cfg = infer.load_config(str(args.root/'assets/songbloom/songbloom_full_150s.yaml'),
                        parent_dir=str(args.root/'assets/songbloom'))
cfg.max_dur = 60
model = load(cfg, dtype=torch.bfloat16, device=torch.device('cpu'))
assert model.diffusion.skeleton_classifier[-1].out_features > 1
print('SONGBLOOM_STRICT_CPU_LOAD_OK', flush=True)
