"""Bounded upstream-kernel audit, explicitly not an end-to-end trainer test.

Load the exact pinned upstream definitions using AST, without substituting a
handwritten loss or importing unrelated Ray/vLLM/image engines. Receipts record
this isolation; passing this test must not be called a framework integration pass.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
import torch


def definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert len(selected) == len(names)
    for node in selected:
        node.decorator_list = []
    source = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *selected], type_ignores=[])
    exec(compile(ast.fix_missing_locations(source), str(path), 'exec'), namespace)


def load(miles, verl):
    namespace = {'torch': torch, 'math': math, 'abc': __import__('abc'), 'Any': Any, 'Optional': Optional}
    definitions(miles / 'miles/backends/fsdp_utils/sde_step_backend.py',
                ['SdeStepBackend', 'FlowSdeStepBackend'], namespace)
    namespace.update({'record_rollout_train_abs_diff': None, 'stack_train_pair_rollout_debug': lambda *a: None})
    definitions(miles / 'miles/backends/fsdp_utils/loss_hub/flow_grpo.py', ['flow_grpo_loss_formula'], namespace)
    namespace['DiffusionLossFn'] = object
    definitions(verl / 'verl_omni/trainer/diffusion/diffusion_algos.py', ['FlowGRPOLoss'], namespace)
    return namespace


class Metrics:
    def __init__(self):
        self.values = {}

    def emit_mean(self, key, *, total, count):
        self.values[key] = float(total.detach() / count)


def compare(namespace, velocity, x, nxt, old, advantages, sigmas, next_sigmas, ref_velocity, sigma_max_value=.9):
    backend = namespace['FlowSdeStepBackend']()
    sigma_max = torch.full_like(sigmas, sigma_max_value)
    schedule = dict(sigmas=sigmas, next_sigmas=next_sigmas, sigma_max=sigma_max, noise_level=.1)
    _, new, mu, std = backend.sde_step_logprob(velocity, x, prev_sample=nxt, **schedule)
    _, _, ref_mu, _ = backend.sde_step_logprob(ref_velocity, x, prev_sample=nxt, **schedule)
    args = SimpleNamespace(diffusion_clip_range=.2, diffusion_noise_level=.1, diffusion_kl_beta=.01)
    ctx = SimpleNamespace(args=args, sde_backend=backend, models={'audio_backbone': None}, device=x.device)
    prepared = SimpleNamespace(latents={'visual': x}, advantage=advantages, component_name='audio_backbone', extras={
        'next_latents': nxt, 'sigmas': sigmas, 'next_sigmas': next_sigmas, 'sigma_max': sigma_max, 'log_prob_old': old})
    metrics = Metrics()
    miles_loss = namespace['flow_grpo_loss_formula'](ctx, [{}, {}], prepared, new_pred={'visual': velocity},
                    old_pred=None, ref_pred={'visual': ref_velocity}, metrics=metrics) / 2
    config = SimpleNamespace(diffusion_loss=SimpleNamespace(adv_clip_max=5., clip_ratio=.2))
    verl_policy, verl_metrics = namespace['FlowGRPOLoss'].compute_loss(old_log_prob=old, log_prob=new,
                                                                     advantages=advantages, config=config)
    dims = tuple(range(1, x.ndim))
    variance = std.square() * (sigmas - next_sigmas).view(-1, *([1] * (x.ndim - 1)))
    exact_kl = ((mu-ref_mu).square() / (2 * variance)).mean(dim=dims).mean()
    # VeRL's FlowGRPOLoss contains policy loss only; its worker must add reference KL.
    verl_with_exact_kl = verl_policy + .01 * exact_kl
    miles_reported_kl = metrics.values['kl_loss']
    return miles_loss, verl_with_exact_kl, {
        'miles_loss': float(miles_loss.detach()), 'verl_policy_loss': float(verl_policy.detach()),
        'verl_plus_exact_transition_kl': float(verl_with_exact_kl.detach()),
        'miles_reported_kl': miles_reported_kl, 'exact_transition_kl_per_dimension': float(exact_kl.detach()),
        'verl_metrics': verl_metrics,
        'transport_note': 'Audio tensor passed through Miles visual key only for isolated kernel audit; not a registered audio pipeline',
        'logprob_note': 'Both audited kernels use per-dimension mean log-density; not joint transition log-density'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--miles', type=Path, required=True); p.add_argument('--verl', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    ns = load(a.miles, a.verl)
    torch.manual_seed(42)
    x = torch.randn(2, 10, 64)
    s, sn = torch.full((2,), .5), torch.full((2,), .4)
    velocity = torch.randn_like(x, requires_grad=True)
    backend = ns['FlowSdeStepBackend']()
    mu, noise, _ = backend.prev_sample_mean_and_std(velocity.detach(), x, s[:, None, None], sn[:, None, None],
                                                   sigma_max=torch.full((2, 1, 1), .9), noise_level=.1)
    nxt = mu + noise * torch.randn_like(x)
    old = backend.log_prob(nxt, mu, noise)
    ref = velocity.detach() + .1
    ml, vl, result = compare(ns, velocity, x, nxt, old, torch.tensor([-1., 1.]), s, sn, ref)
    gm = torch.autograd.grad(ml, velocity, retain_graph=True)[0]
    gv = torch.autograd.grad(vl, velocity)[0]
    result.update({'scope': 'Isolated exact upstream loss/SDE kernels, CPU analytic fixture; no model training or perceptual result',
                   'finite_gradients': bool(torch.isfinite(gm).all() and torch.isfinite(gv).all()),
                   'gradient_norms': [float(gm.norm()), float(gv.norm())],
                   'expected_kl_factor_for_dt_point1': 10.,
                   'observed_exact_to_miles_kl_ratio': result['exact_transition_kl_per_dimension'] / result['miles_reported_kl'],
                   'sources': {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for root, path in [(a.miles, a.miles/'miles/backends/fsdp_utils/sde_step_backend.py'),
                                          (a.miles, a.miles/'miles/backends/fsdp_utils/loss_hub/flow_grpo.py'),
                                          (a.verl, a.verl/'verl_omni/trainer/diffusion/diffusion_algos.py')]}})
    assert abs(result['observed_exact_to_miles_kl_ratio']-10) < 1e-4
    assert result['finite_gradients']
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
