"""Explicit linear-module LoRA for native, non-HF diffusion backbones."""
from contextlib import contextmanager
import math
import torch
from torch import nn


class LinearLoRA(nn.Module):
    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = alpha / rank
        self.enabled = True

    def forward(self, x):
        value = self.base(x)
        if self.enabled:
            delta = torch.nn.functional.linear(torch.nn.functional.linear(x.float(), self.a), self.b)
            value = value + (delta * self.scale).to(value.dtype)
        return value


def install(root, prefixes):
    root.requires_grad_(False)
    targets = [(name, module) for name, module in root.named_modules()
               if isinstance(module, nn.Linear) and any(name.startswith(p) for p in prefixes)]
    if not targets:
        raise ValueError('No backbone linear modules matched')
    for name, module in targets:
        owner, _, leaf = name.rpartition('.')
        setattr(root.get_submodule(owner) if owner else root, leaf, LinearLoRA(module))
    return [name for name, _ in targets]


@contextmanager
def reference(root):
    layers = [m for m in root.modules() if isinstance(m, LinearLoRA)]
    previous = [m.enabled for m in layers]
    try:
        for m in layers:
            m.enabled = False
        yield
    finally:
        for m, enabled in zip(layers, previous):
            m.enabled = enabled
