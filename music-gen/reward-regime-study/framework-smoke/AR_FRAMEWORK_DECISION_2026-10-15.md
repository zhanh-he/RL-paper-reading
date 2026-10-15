# AR Framework Decision: TRL

**Use TRL for YuE2, Muse and HeartMuLa discrete-token policy updates.** Do not
start a Miles-LLM AR migration or a parallel framework-comparison training arm.
The continuous-flow decision remains Miles-Diffusion for DiffRhythm2/SongBloom.
Miles-LLM and Miles-Diffusion are distinct implementations, not interchangeable
feature matrices.

Public reading page: [TRL / Miles / VeRL](https://zhanh-he.github.io/RL-paper-reading/demos/frameworks.html).

## Source-Backed Reason

The inspected Miles-LLM revision is
`cf91d625b3074f480b31ddcfd6604298407899f3` (2026-10-15 review).
It does support HF-native FSDP2 training, not just Megatron. However, its
**LoRA training requires the Megatron backend**. The LoRA argument validator
explicitly rejects FSDP; the backend matrix also lists no LoRA for torchtitan.
The production LoRA path combines a Megatron/Bridge provider, adapter name/export
mapping and SGLang serving support. This is an additional port/validation surface
for our native audio policies, not an assertion that Miles cannot train them.

In contrast, TRL + PEFT can retain a compatible HF causal-LM actor and a
project-local native sampler/log-prob adapter, without requiring a Megatron
conversion or an SGLang serving port. It supports native generation or optional
vLLM, explicit reference KL, gradient accumulation/checkpointing and Trainer
checkpoint/resume. These features still need model-specific acceptance.

## Stability, Speed And Memory

| Criterion | Judgment for our study |
|---|---|
| Stability / correctness | Prefer TRL because the native codec-token support, EOS schedule, temperature, masks and reference likelihood can be audited with one actor implementation. This is an integration-risk judgment, not a measured framework failure-rate ranking. Miles' TITO, consistency diagnostics and recovery are real advantages, not absent features. |
| Speed | No matched YuE2/Muse TRL-versus-Miles throughput measurement exists. Miles' SGLang/async/distributed optimizations may be valuable at larger scale. Our small dense actors, G2 and scarce usually single-GPU allocations do not justify assuming that those gains exceed port, sync and offload costs. Keep Muse's existing vLLM path where validated; TRL does not mean forcing slow HF-only generation. |
| Memory | TRL + PEFT can use the frozen base via adapter disabling instead of a permanent second full reference actor, when that base really is the selected reference. Native rollout avoids a separate always-resident serving replica. Codec/scorer still require phase offload; activation/KV memory and audio duration remain the real constraints. This is not a claim that every TRL layout uses less memory than every Miles layout. |

Do not rely on framework defaults: audited TRL0.24 has `beta=0`,
`num_iterations=1`, and `loss_type=dapo` by default. Set beta0.01 and the intended
loss, schedule, normalization and update count explicitly in a new run manifest.
A single update from the old policy has ratio1 at that gradient, so PPO clipping
is not an effective extra bound there. Switching trainer names does not fix that.
Sampled k3 KL is not exact full-vocabulary KL or a hard trust-region guarantee.

## Model-Specific Status And Acceptance

- **YuE2:** existing formal results remain PyTorch + PEFT. A future TRL port must
  keep the native semantic prefix, codec+EOS action support, minimum-token EOS
  mask, temperature/CFG and terminal/truncation semantics identical in sampling,
  old/current log-probs and reference scoring. Plain full-vocabulary teacher
  forcing is not automatically the same policy. Record actual sampling log-probs
  and test cached/full-context agreement. No full YuE2 TRL acceptance is claimed.
- **Muse:** existing integration is ms-swift/TRL + vLLM with the native codec
  and a model-specific multi-turn scheduler. Preserve completion token IDs,
  per-turn masks, audio boundary symbols and frozen decoder randomness. Audit
  vLLM-versus-trainer log-probs and adapter weight version. Keep the existing
  pinned environment until a replacement passes acceptance; do not force it onto
  the HeartMuLa smoke's TRL version merely to unify package names.
- **HeartMuLa:** actual GRPOTrainer0.24 two-update backbone test passed on5090,
  all8 codebooks and217 global/local attention/MLP targets. FP32,4s budget,
  G2, peak allocated26.025GiB, receipt elapsed86.804s excluding cold actor load.
  This is a bounded compatibility result, not a YuE2/Muse benchmark or a formal
  long-audio/distributed success. The accepted runner remains frozen.

The accepted HeartMuLa baseline pins TRL0.24.0,
`04fd1203af4fc1e629f58e0ac0d0c5bb95f82a45`. Current TRL documentation has newer
experimental rollout APIs; do not silently treat them as tested0.24 interfaces.
Record each actual environment/version and all subclass overrides in its receipt.
Before formal submission: two real optimizer updates, finite attention/MLP
gradients, matched likelihood support, reference identity/drift, save/restart
including optimizer/RNG/data cursor, and intended-duration memory checks.

This task changes documentation and framework selection only. No new training,
scheduled task, result relabeling, dataset change or checkpoint restart is made.

## Primary Sources And Local Evidence

- [Miles-LLM backend matrix](https://github.com/radixark/miles/blob/cf91d625b3074f480b31ddcfd6604298407899f3/docs/user-guide/training-backend.md).
- [Miles-LLM LoRA contract](https://github.com/radixark/miles/blob/cf91d625b3074f480b31ddcfd6604298407899f3/docs/advanced/lora.md) and [backend validation](https://github.com/radixark/miles/blob/cf91d625b3074f480b31ddcfd6604298407899f3/miles/utils/lora/arguments.py).
- [TRL0.24 GRPO documentation](https://huggingface.co/docs/trl/v0.24.0/grpo_trainer), [configuration](https://github.com/huggingface/trl/blob/04fd1203af4fc1e629f58e0ac0d0c5bb95f82a45/trl/trainer/grpo_config.py), [trainer](https://github.com/huggingface/trl/blob/04fd1203af4fc1e629f58e0ac0d0c5bb95f82a45/trl/trainer/grpo_trainer.py).
- [HeartMuLa receipt](heartmula-trl-receipt.json), [accepted runner](heartmula-accepted-runner.py), [existing YuE2 implementation](../train_yue2.py), [existing Muse reward/scheduler adapter](../muse_plugin.py).
- [Selected Flow framework and numerical corrections](FLOW_FRAMEWORK_DECISION_2026-10-15.md).
