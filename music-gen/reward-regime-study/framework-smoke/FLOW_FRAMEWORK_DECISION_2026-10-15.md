# Flow-GRPO Framework Decision

**Decision: Miles-Diffusion, synchronous Flow-GRPO, FSDP2, backbone LoRA.**
Discrete-token policies stay on TRL. DiffRhythm2 and SongBloom share the Miles
route. VeRL-Omni is not an active fallback, a parallel implementation, or a
preregistered framework-comparison arm. Historical audits remain as evidence.

This is a source-backed engineering choice for our native audio models and
scarce, usually single-GPU allocations. It is **not a measured claim that Miles
is universally faster, more stable, or smaller than VeRL-Omni**. Neither full
Flow framework has passed our real-model distributed lifecycle acceptance yet.
This decision does not start another formal training job.

## Compared Revisions

- Miles-Diffusion: `66f61901873a51be1de2072998090ac754fb18f3`.
- VeRL-Omni: `f31a6116119fd62a89554987f51ae279a567d66b`.
- Actual prior TRL smoke: v0.24.0, not an unspecified latest version.

## Stability, Speed And Memory

| Criterion | Miles-Diffusion | VeRL-Omni | Decision for this project |
|---|---|---|---|
| Integration / numerical stability | Explicit native model backend, batch, SDE, loss and rollout hooks. Its synchronous entry point makes generate/update/sync ordering easy to inspect. Default rollout still starts SGLang engines; an audio worker integration is required. | Documented non-diffusers training extension, FSDP2, LoRA, resume tests and rollout-consistency diagnostics. Its documented online path requires a vLLM-Omni pipeline; our two native audio pipelines need a port. Default v1 uses TransferQueue / ReplayBuffer, including in sync mode. | Miles: prefer narrower orchestration and preserving native audio forwards. Lower migration risk is our engineering judgment, not a measured failure-rate difference. Both need acceptance. |
| End-to-end speed | LoRA-only IPC/NCCL sync and sample/timestep microbatch controls exist. Custom rollout hooks permit retaining native block scheduling. No matched music throughput benchmark exists. | Continuous/request batching, async reward serving and rollout correction can improve supported recipes. Authors report about25% above the diffusers Flow-GRPO reference on Qwen-Image, not above Miles and not on these audio models. | No demonstrated numerical winner. Choose Miles to avoid building a second full inference-engine port and orchestration route. Time-to-correct-first-run matters more than unverified image-to-audio speed extrapolation. |
| GPU memory | FSDP2, LoRA, activation checkpointing, phase offload and `ref_mode=lora_base` support. The latter evaluates the frozen base without another reference-model copy. | FSDP2, LoRA, checkpointing, parameter/optimizer offload and reference-in-actor support too. vLLM memory reservation is tunable, not proof of an intrinsically larger footprint. | No demonstrated numerical winner. Use shared-base reference and phase-exclusive residency. A one-GPU FSDP setup does not shard weights across GPUs. Framework branding does not remove activation/trajectory memory. |

Do not infer maturity from stars or use missing optional dependencies as evidence
of algorithmic instability. Both have CPU/GPU/E2E test infrastructure, but neither
published recipe matrix certifies our DiffRhythm2 or SongBloom port.

### What We Actually Measured

| Native backbone | Bounded test | Peak PyTorch allocated memory | Timed section |
|---|---|---:|---:|
| SongBloom, all12 acoustic DiT blocks | G2,4.48s audio, two updates per isolated loss | 7.081GiB | 8.963s |
| DiffRhythm2, all16 DiT blocks | G2,about4s audio, two updates per isolated loss | 16.274GiB | 7.556s |

See the [native smoke receipts](README.md#real-flow-backbone-smokes). These are
combined native-model/kernel-audit scopes, **not two framework timing results**.
They exclude cold model/frontend loading, Ray/engine lifecycle, full trajectory
training and formal30-60s replay inference. They cannot establish long-rollout
fit, total device memory, or Miles-versus-VeRL memory/speed ratios. No such ratio
or promised formal steps/hour is reported.

## Two Corrections To The Earlier Comparison

**Dependencies are not a Miles advantage.** Miles' complete requirements also
pin Torch2.11 / Transformers5.12.1 / a development diffusers commit. The inspected
VeRL-Omni stack pins Transformers5.13-5.14.1 / vLLM0.30 / upstream Git revisions.
Both conflict with at least some native4.x audio environments. Use an isolated,
pinned training environment and explicitly tested native worker boundaries; do
not upgrade shared inference/reward environments. A minimal dependency subset
would itself need import/lifecycle acceptance, not an assumed compatibility pass.

**The default KL scale caveat applies to both frameworks.** The older real-model
audit used VeRL's policy loss plus an explicitly added exact conditional KL. That
was not VeRL's default worker KL, so its numerical difference from Miles was not
evidence that VeRL had a better default. The new
[source-SHA-verified CPU audit](framework-selection-kl.json) loads both exact
upstream definitions and finds:

| Flow-SDE step magnitude | Miles default KL / dimension | VeRL default KL / dimension | Exact discrete-transition KL / dimension |
|---|---:|---:|---:|
| 0.1 | 0.005050126 | 0.005050126 | 0.050501250 |
| 0.01 | 0.000050501 | 0.000050501 | 0.005050109 |

Identity checks are0 and gradients are finite. The default denominators use the
diffusion scale squared, while discrete-transition variance includes `abs(dt)`.
This can define a rate/time-weighted regularizer rather than an implementation
error. It must not be called exact discrete-transition KL. Choosing another
framework does not resolve this convention automatically.

For our accepted audio port, use the explicitly named frozen-reference
**conditional Gaussian KL per active latent dimension**, with the actual
recorded transition variance, and also retain the upstream rate-scaled diagnostic
under a different name. Exclude padding and deterministic zero-variance steps.
Keep beta0.01 as the proposed initial setting, not as an assertion of equivalence
with an older rate-scaled beta. Both policy kernels use dimension-mean log-density
ratios, not joint Gaussian density ratios; disclose that normalization separately.

## Single Implementation Route

1. **Miles owns training.** Native model backend + FSDP2 actor + AdamW + framework
   checkpoint/optimizer lifecycle. All DiT attention/MLP LoRA targets; no head-only
   substitution. SongBloom's AR planner, conditioners and VAE remain frozen.
2. **Native audio rollout worker.** Preserve the tested SDE/CFG/block-history
   semantics. Integrate at engine creation, offload and adapter-sync boundaries.
   Miles currently hardcodes SGLangDiffusionEngine at startup: the rollout flag
   alone does not replace this factory. This is a required project extension,
   not an existing working audio plugin. Do not bypass it with `--train-only`,
   which disables online weight sync and framework evaluation.
3. **Synchronous, on-policy first.** No async stale-policy batches, replay
   retention or speculative work. Record actual rollout old log-probs and policy
   version; verify current-actor recomputation before each optimizer update.
   DiffRhythm2 rebuilds clean history with current parameters for training.
4. **Memory before concurrency.** G2, microbatch1 / accumulation, checkpoint
   DiT blocks; store selected stochastic transitions on CPU in FP32. Record the
   sampling-window rule and use the same window within a reward group. Offload
   codec/scorer/rollout by phase and use a shared frozen base for reference. Do
   not introduce another permanent GPU-sized reference or old-policy replica.
   LoRA-only sync must be validated, not assumed from SGLang's implementation.
5. **Fixed study.** Existing CMI270/30 split, LR3e-4, existing reward/guard versions,
   native conditioning and frozen replay prompts stay unchanged. Bounded4s
   acceptance does not silently redefine the formal training duration. Public
   replay remains30-60s MP3, with private lossless scoring originals.

Before formal submission, the selected Miles route must pass a real framework
generate -> score -> update -> sync -> regenerate cycle, two nonzero backbone
updates, identity/reference drift checks, save/restart with optimizer/RNG/data
cursor restored, and memory/time profiling at the intended training budget.
Record peak allocated **and reserved**, whole-device memory, rollout/decode/reward/
update/sync times and policy version. A failed required gate stops the job; it
does not relabel the old PyTorch/kernel smoke as a Miles trainer pass. Only Miles
receives further integration effort. No VeRL comparison job is planned.

## Primary Source Pointers

- [Miles customization hooks](https://github.com/radixark/miles_diffusion/blob/66f61901873a51be1de2072998090ac754fb18f3/docs/user-guide/customization.md), [engine factory](https://github.com/radixark/miles_diffusion/blob/66f61901873a51be1de2072998090ac754fb18f3/miles/ray/rollout.py), [synchronous loop](https://github.com/radixark/miles_diffusion/blob/66f61901873a51be1de2072998090ac754fb18f3/train_diffusion.py).
- [Miles actor/reference/offload](https://github.com/radixark/miles_diffusion/blob/66f61901873a51be1de2072998090ac754fb18f3/miles/backends/fsdp_utils/actor.py), [LoRA sync](https://github.com/radixark/miles_diffusion/blob/66f61901873a51be1de2072998090ac754fb18f3/docs/advanced/lora.md), [requirements](https://github.com/radixark/miles_diffusion/blob/66f61901873a51be1de2072998090ac754fb18f3/requirements.txt).
- [VeRL non-diffusers integration](https://github.com/verl-project/verl-omni/blob/f31a6116119fd62a89554987f51ae279a567d66b/docs/contributing/integrating_a_non_diffusers_model.md), [performance/memory tuning](https://github.com/verl-project/verl-omni/blob/f31a6116119fd62a89554987f51ae279a567d66b/docs/perf/tuning_guide.md), [throughput claim and scope](https://github.com/verl-project/verl-omni/blob/f31a6116119fd62a89554987f51ae279a567d66b/README.md).
- [VeRL default KL](https://github.com/verl-project/verl-omni/blob/f31a6116119fd62a89554987f51ae279a567d66b/verl_omni/trainer/diffusion/diffusion_algos.py), [SDE scale definition](https://github.com/verl-project/verl-omni/blob/f31a6116119fd62a89554987f51ae279a567d66b/verl_omni/pipelines/schedulers/flow_match_sde.py), [dependencies](https://github.com/verl-project/verl-omni/blob/f31a6116119fd62a89554987f51ae279a567d66b/pyproject.toml).
