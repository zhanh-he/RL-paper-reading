# Backbone LoRA / Framework Acceptance

2026-10-15. This directory contains bounded compatibility work, not another
formal training run. No scheduled task or automatic monitor is used.

## Decision Boundary

- Discrete AR: TRL is selected. HeartMuLa needs a native eight-codebook adapter;
  it is not an ordinary single-stream causal LM. YuE2's existing results remain
  **PyTorch + PEFT**, not retroactively relabeled TRL.
- DiffRhythm2: block-autoregressive **continuous flow** policy. Adapt all DiT
  attention/MLP blocks, with Flow-GRPO, not token GRPO. Recompute text and clean
  history with current actor parameters; a rollout KV cache is not a training
  graph.
- SongBloom: adapt the acoustic `nar_dit` backbone. Keep the AR sketch planner,
  conditioning, projection heads and VAE frozen. Native input is reference audio
  plus lyrics, not an unsupported text-count prompt.
- A two-update test permits a framework decision, **not** a new 270-step formal
  launch. Rank8 / alpha16 / LR3e-4 / G2 / beta0.01 are smoke settings. The formal
  frozen 270-train / 30-valid split remains unchanged; WSB stays sealed.

## Pinned Sources

| Route | Revision | Scope |
|---|---|---|
| [TRL 0.24.0](https://github.com/huggingface/trl/tree/04fd1203af4fc1e629f58e0ac0d0c5bb95f82a45) | `04fd1203af4fc1e629f58e0ac0d0c5bb95f82a45` | Actual GRPOTrainer with native generation / teacher-forcing overrides |
| [Miles-Diffusion](https://github.com/radixark/miles_diffusion/tree/66f61901873a51be1de2072998090ac754fb18f3) | `66f61901873a51be1de2072998090ac754fb18f3` | Isolated exact upstream SDE/loss kernels; full actor integration is a separate gate |
| [VeRL-Omni](https://github.com/verl-project/verl-omni/tree/f31a6116119fd62a89554987f51ae279a567d66b) | `f31a6116119fd62a89554987f51ae279a567d66b` | Isolated exact upstream FlowGRPOLoss; reference KL is outside this policy kernel |

The isolated loss test loads the pinned source definitions with AST, rather than
substituting a handwritten implementation. It intentionally does not import the
Ray/vLLM/image stack. This is **not** a full framework integration pass.

## Measured Kernel Finding

### Real TRL Model Smoke

[HeartMuLa receipt](heartmula-trl-receipt.json): **two actual GRPOTrainer optimizer
updates passed on RTX5090**, with all8 codebooks in the loss and217 attention/MLP
linear targets across the global and local Transformers. Heads/codec stay frozen.
Global/local LoRA tensors had nonzero gradients at both updates. Peak allocated
memory was26.025GiB; trainer time86.44s, excluding cold checkpoint load. Four-second
rollouts, G2, one real CMI train condition, frozen real SP-Richness scorer and
audio SHA checks were used. Reference k3 KL was0 at update1 and0.00164128 at update2.
These are runtime checks, not evidence of musical improvement or full validation.

[Group0](heartmula-group-0.json) / [group1](heartmula-group-1.json) retain exact
score/audio SHA bindings. The largest measured cached-sampling versus full
teacher-forcing token log-probability error across the four FP32 rollouts
was0.000067115.

The measured smoke used initial LR3e-4 with TRL's two-step linear schedule
(second update1.5e-4). It is not a fixed-LR formal experiment. FP32 was used for
the cache/full-sequence likelihood audit; FP16/BF16 discrepancies were caught,
not bypassed. FP32 initially OOMed when generator+codec shared the GPU and when
keeping rollout KV during backward. Explicit offload, KV release, per-layer
activation checkpointing and microbatch1/accumulation2 resolved that on32GB.

The [accepted runner](heartmula-accepted-runner.py) is frozen alongside its
receipt. A future native mixed-precision/distributed path still needs its own
acceptance; this successful FP32 single-GPU adapter does not establish that port.

### Real Flow Backbone Smokes

| Native model | Backbone linear targets | Peak allocated GPU | Rollout/update time | Result |
|---|---:|---:|---:|---|
| [SongBloom](songbloom-flow-receipt.json) | 60, all12 acoustic DiT attention/MLP blocks | 7.081GiB | 8.96s | Two updates per isolated Miles / VeRL loss |
| [DiffRhythm2](diffrhythm2-flow-receipt.json) | 112, all16 DiT attention/MLP blocks | 16.274GiB | 7.56s | Two updates per isolated Miles / VeRL loss |

Times exclude checkpoint/frontend loading. Both start from identical adapters
and the same real G2 audio within each model. Real SongBench-Instrument rewards
were used: SongBloom0.468764/0.456134, DiffRhythm2 0.339226/0.361693. These numbers
are smoke-group values, **not validation means**. Attention and MLP gradients are
finite/nonzero; adapter parameters change; identity reference KL is0, and exact
conditional reference KL becomes nonzero after the first update. DiffRhythm2 also
checks a second-block transition with rebuilt clean history; its FP16 native
cache/full-context maximum velocity difference is0.0234375, not exact equality.

The first short SongBloom group had tied SP-Richness rewards and was correctly
rejected as a policy-gradient acceptance test. The successful rerun explicitly
uses Instrument instead; no artificial rewards or forced advantages were added.
This does not prove long-form Richness training is impossible. Its native short
VAE chunking error is avoided by decoding the entire short latent, without
modifying the shared vendor checkout.

Neither flow test runs a Ray/FSDP actor or SGLang/vLLM-Omni rollout engine. Thus
these times and memory **cannot rank whole-framework throughput**. Native model
trainability is established at this bounded scope; framework gate7 remains open.

### Recommended Next Port

Keep **TRL for AR**. For the first full Flow framework port, prioritize
**Miles-Diffusion's native FSDP backend and custom rollout hooks**, retaining
VeRL-Omni as the second route. This is an engineering priority, not a benchmark
victory: Miles exposes model-backend, batch, rollout and SDE hooks, so the native
audio forward paths tested here can be retained. Its default visual transport
and scale-normalized KL still need explicit audio adapters and acceptance.
VeRL-Omni's current full stack requires a separate Transformers5.x/vLLM0.30
environment and a registered native audio rollout/pipeline; forcing that stack
into the existing4.x generator environments is not an acceptable shortcut.

The [5090](import-receipt-5090.json) and [Gadi](import-receipt-gadi.json) import
probes record incomplete optional dependencies. Missing `wandb` or `tensordict`
is an environment setup result, **not proof a framework cannot train audio**.
No complete distributed-framework success is claimed, and no new formal training
has been launched. The next decision is whether to finish the Miles audio port
first or fund both complete framework paths before formal runs.

### Flow Kernel Semantics

The [raw receipt](flow-kernel-receipt.json) uses a clearly labeled CPU analytic
fixture, not a music quality experiment. Both upstream loss paths produce finite
gradients. For identical Gaussian transition variance and `delta_t=0.1`:

| Quantity | Measured |
|---|---:|
| Miles reported KL, per dimension | 0.005050126 |
| Exact one-transition Gaussian KL, per dimension | 0.050501250 |
| Ratio | 9.999998 |

Miles' current formula divides by the diffusion-scale square, not the complete
transition standard deviation square including `delta_t`. That can represent a
different rate/time-weighted regularizer; it must **not** be labeled identical to
our exact one-step conditional KL or compared at equal beta without qualification.
Both audited policy kernels use mean log-density, not joint latent log-density.
VeRL's `actor/ppo_kl` is old/current-policy drift, not a frozen-reference KL.

Full VeRL-Omni currently pins Transformers5.13-5.14 and vLLM0.30, conflicting with
the models' native Transformers4.x environments. Miles' current Flow-GRPO batch
path hardcodes the `visual` stream. Neither is a ready-made audio model port.
Do not upgrade the shared reward or native inference environments to hide this.

## Acceptance Gates

1. Real source import and recorded dependency failures.
2. Native checkpoint strict load; explicit trainable module list excluding heads.
3. Real G2 audio rollouts, scored PCM24-FLAC SHA binding and non-tied rewards.
4. Native cached rollout versus current-parameter full-context likelihood/velocity.
5. Finite nonzero attention **and MLP** gradients, two optimizer updates, adapter
   parameter changes; all eight HeartMuLa codebooks included.
6. Identity reference check and reference drift after update, with KL units named.
7. Framework-owned distributed rollout/worker/optimizer/checkpoint cycle.

`smoke_heartmula_trl.py` exercises an actual GRPOTrainer. The two flow scripts
exercise real model backbones and exact isolated framework kernels, **not gate7**.
Passing an import, CPU analytic test, or isolated loss kernel never means all gates
passed. Short smoke audio is not inserted into the public 30-60-second demos.

## Preserved Old Results

The old output-head runs are **deprecated head-only ablations, to be replaced**.
They cannot support claims that full-model RL succeeds or fails:

| Model / target | Verified updates | Full validation | Disposition |
|---|---:|---|---|
| DiffRhythm2 / richness | 107 | 30/30 at0/50/100 | G2P `Unknown language: other`; failed, not270 complete |
| HeartMuLa / richness | 36 | None | Stopped; all journaled checkpoint/audio SHA verified |
| HeartMuLa / instrument | 31 | None | Stopped; all journaled checkpoint/audio SHA verified |
| HeartMuLa / arrangement | 31 | None | Stopped; all journaled checkpoint/audio SHA verified |
| SongBloom / richness | 4 | None | Short-latent chunked VAE decode failure |
| SongBloom / instrument | 4 | None | Same failure |

Other new head arms did not start and have been cancelled. Sources, actual steps,
checkpoints on the experiment machines, audio originals and receipts are preserved.
Partial training-group rewards are curves only, never final validation means.

YuE2 inference recovery has five conditions per checkpoint: three original
validation examples plus two explicitly labeled instrument-count probes sharing
lyrics and seed. Five frozen baselines +45 checkpoint outputs are generated,
scored on private lossless originals and exported only as MP3 listening copies.
These probes do not enlarge the30-condition validation set or the test set.

DiffRhythm2's existing0/50/100 full-validation examples0/10/20 are also recovered
as nine60-second MP3s, explicitly labeled deprecated head-only audio, not new
backbone inference or a270-step result.

## Cluster Resource Receipt

Interactive requests were attempted on Gadi iv96 Hopper181290527, then wa66
Volta181291096, and Kaya145738. None received a GPU during this bounded session.
Gadi reported resource/reservation constraints; Kaya's estimated start moved to
16:12 AWST (an estimate, not a promise). Pending interactive requests were cancelled
to avoid an unattended allocated shell. No unrelated jobs were interrupted.

Gadi and Kaya both ran the small CPU analytic kernel audit with identical results:
[Gadi receipt](flow-kernel-receipt-gadi.json),
[Kaya receipt](flow-kernel-receipt-kaya.json). These are CPU unit checks, not GPU
training. All real model smokes ran on5090 **after** completing the existing YuE2
inference/public scoring work. No scheduled task or monitoring process is left.
