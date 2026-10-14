"""Restricted output-head GRPO; frozen upstream features, not backbone RL."""
import math
import torch
from torch import nn
from torch.distributions import Categorical


class HeadPolicy(nn.Module):
    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = alpha/rank
        self.active = True
        self.capture = False
        self.features = []

    def logits(self, x, reference=False):
        with torch.autocast(device_type=x.device.type, enabled=False):
            out = self.base(x.to(self.base.weight.dtype)).float()
            if self.active and not reference:
                out = out + (x.float() @ self.a.t() @ self.b.t())*self.scale
        return out

    def forward(self, x):
        if self.capture: self.features.append(x.detach().cpu().clone())
        return self.logits(x).to(x.dtype)

    def guided(self, features, cfg, reference=False):
        logits = self.logits(features, reference=reference)
        return logits[1] + cfg*(logits[0]-logits[1]) if logits.shape[0] == 2 else logits[0]

    def discrete_action(self, cfg):
        features = self.features.pop()
        x = features.to(self.a.device)
        distribution = Categorical(logits=self.guided(x, cfg).reshape(-1))
        action = distribution.sample()
        return action, {'features': features, 'action': action.cpu(),
                        'old_logprob': distribution.log_prob(action).detach().cpu(), 'cfg': cfg}

    def discrete_terms(self, record):
        x = record['features'].to(self.a.device).clone()
        p = Categorical(logits=self.guided(x, record['cfg']).reshape(-1))
        q = Categorical(logits=self.guided(x, record['cfg'], reference=True).reshape(-1))
        return p.log_prob(record['action'].to(self.a.device)), torch.distributions.kl_divergence(p, q)

    def adapter_state(self):
        return {'a': self.a.detach().cpu(), 'b': self.b.detach().cpu(), 'scale': self.scale}

    def restore(self, state):
        if state['scale'] != self.scale: raise ValueError('Adapter scaling changed')
        with torch.no_grad(): self.a.copy_(state['a']); self.b.copy_(state['b'])
