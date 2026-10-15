"""CPU-only source audit for the framework decision, not a trainer benchmark."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch

from audit_flow_kernels import compare, definitions, load


def revision(root):
    if not (root / ".git").exists():
        return None
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def run(miles, verl):
    pins = {
        "miles": "66f61901873a51be1de2072998090ac754fb18f3",
        "verl": "f31a6116119fd62a89554987f51ae279a567d66b",
    }
    actual = {"miles": revision(miles), "verl": revision(verl)}
    for name, value in actual.items():
        assert value is None or value == pins[name], actual
    expected = {
        "miles/backends/fsdp_utils/sde_step_backend.py": "a4c199e868192c00c5f25ee4a8478d04d78e296e6c2b4cbe7d5fe9aaff63bbc2",
        "miles/backends/fsdp_utils/loss_hub/flow_grpo.py": "9d67d7b451f8b0625f72e7c598e8ea93ffee73cb7516fc9d8ebeaed1a01f2dbf",
        "verl_omni/trainer/diffusion/diffusion_algos.py": "1411e6facf4ceee0ef37f3d405fc2f983364bc7488f8867924c6fcf86c5d78a6",
        "verl_omni/pipelines/schedulers/flow_match_sde.py": "d3041878eb111790f4f55acedb9409fd434633898265d83d072f06f82b621589",
    }
    for relative, expected_hash in expected.items():
        root = miles if relative.startswith("miles/") else verl
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected_hash
    namespace = load(miles, verl)
    verl_loss = verl / "verl_omni/trainer/diffusion/diffusion_algos.py"
    definitions(verl_loss, ["KLLoss"], namespace)
    backend = namespace["FlowSdeStepBackend"]()
    cases = []
    for dt in [0.1, 0.01]:
        for reference_shift in [0.0, 0.1]:
            torch.manual_seed(42)
            x = torch.randn(2, 10, 64)
            velocity = torch.randn_like(x, requires_grad=True)
            s = torch.full((2,), 0.5)
            sn = s - dt
            sigma_max = torch.full_like(s, 0.9)
            schedule = dict(
                sigmas=s, next_sigmas=sn, sigma_max=sigma_max, noise_level=0.1
            )
            mu, transition_std, _ = backend.prev_sample_mean_and_std(
                velocity.detach(), x, s[:, None, None], sn[:, None, None],
                sigma_max=sigma_max[:, None, None], noise_level=0.1,
            )
            nxt = mu + transition_std * torch.randn_like(x)
            old = backend.log_prob(nxt, mu, transition_std)
            ref = velocity.detach() + reference_shift
            _, _, actor_mu, diffusion_scale = backend.sde_step_logprob(
                velocity, x, prev_sample=nxt, **schedule
            )
            _, _, ref_mu, _ = backend.sde_step_logprob(
                ref, x, prev_sample=nxt, **schedule
            )
            ml, _, result = compare(
                namespace, velocity, x, nxt, old, torch.tensor([-1.0, 1.0]),
                s, sn, ref,
            )
            vl_kl, _ = namespace["KLLoss"].compute_loss(
                prev_sample_mean=actor_mu, ref_prev_sample_mean=ref_mu,
                std_dev_t=diffusion_scale,
            )
            ml_grad = torch.autograd.grad(ml, velocity, retain_graph=True)[0]
            vl_grad = torch.autograd.grad(vl_kl, velocity)[0]
            reported = float(vl_kl.detach())
            exact = result["exact_transition_kl_per_dimension"]
            assert torch.isfinite(ml_grad).all() and torch.isfinite(vl_grad).all()
            assert abs(reported - result["miles_reported_kl"]) < 1e-8
            ratio = exact / reported if reported else None
            if reference_shift == 0:
                assert exact == reported == 0.0
            else:
                assert abs(ratio - 1.0 / dt) < 2e-3, ratio
                assert vl_grad.norm() > 0
            cases.append({
                "dt_magnitude": dt,
                "reference_velocity_shift": reference_shift,
                "miles_reported_kl": result["miles_reported_kl"],
                "verl_default_kl": reported,
                "exact_transition_kl_per_dimension": exact,
                "exact_to_default_ratio": ratio,
                "finite_gradients": True,
            })
    paths = [
        (miles, "miles/backends/fsdp_utils/sde_step_backend.py"),
        (miles, "miles/backends/fsdp_utils/loss_hub/flow_grpo.py"),
        (verl, "verl_omni/trainer/diffusion/diffusion_algos.py"),
        (verl, "verl_omni/pipelines/schedulers/flow_match_sde.py"),
    ]
    return {
        "scope": "CPU analytic fixture, exact AST-isolated upstream kernels; no full framework import, GPU training, speed or memory comparison",
        "pins": pins,
        "git_metadata_present": {name: value is not None for name, value in actual.items()},
        "pinned_source_sha256_verified": True,
        "torch": torch.__version__,
        "cases": cases,
        "interpretation": "Both default Flow-SDE KL losses divide by diffusion scale squared, not full transition variance including abs(dt). This can be a rate-weighted regularizer; it is not exact discrete-transition KL. No stability winner follows from this test.",
        "sources": {
            str(root.name) + "/" + relative:
                hashlib.sha256((root / relative).read_bytes()).hexdigest()
            for root, relative in paths
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--miles", type=Path, required=True)
    parser.add_argument("--verl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.miles, args.verl)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
