# Backbone RL Round: 2026-10-15

## Verified Recovery, 2026-10-20

The launch table below is a historical snapshot, not current scheduler status.
Diff's three arms all completed270 updates with full30-condition validation
at0/50/100/270 and verified score/audio/checkpoint bindings. Muse/Heart's six
jobs failed before training on missing `generate_objective` during scorer
bootstrap; none remains queued. [Recovery and five-example MP3 replays](../runs/2026-10-20-songbench-review/README.md).
The user confirmed a new pure seven-head mean reward, but its
[execution matrix](../SONGBENCH_MEAN7_PRIORITY_PLAN_2026-10-20.md) awaits approval.
Do not relabel these guarded single-head/SP runs as Mean7 experiments.

This is the newly authorized backbone round, not a relabeling of old head-only
ablations or the earlier YuE2 SongEval mock sweep.

## Fixed Hyperparameters

| Setting | New round |
| --- | --- |
| Learning rate | **3e-4, constant**, same as the current YuE2 SP/AI study |
| Reference coefficient | **beta=0.05**, explicitly enabled |
| Group | Two independent candidates for the same condition |
| Adapter | Rank8 / alpha16, attention + MLP backbone; frozen codec |
| Rewards | SP Richness / SongBench-Instrument / SongBench-Arrangement, separate arms |
| Data | Frozen CMI projection,270 train /30 validation; no replacement subset |
| Test | WildSongBench192 sealed; not used to select checkpoints or hyperparameters |
| Duration | DiffRhythm2 / HeartMuLa native60s budgets; Muse training max512 tokens |
| Seed |42; fixed validation seed5101+index |
| Public listening files | MP3 only; private scored PCM24-FLAC with separate hashes |

The learning rate is a controlled setting, not a guarantee of equal update size
across architectures. Equal beta is also not equal regularization strength:
AR uses sampled k3 KL per valid audio token; Flow uses exact conditional
Gaussian transition KL per active latent dimension. Neither is a hard KL cap.
Flow uses dimension-mean log-density ratios, not joint Gaussian density ratios.
Historical beta0.01 curves retain their original labels and implementation.

## Actual Launches

Execution priority: **DiffRhythm2, Muse, HeartMuLa, ACE-Step1.5, other models**.
The private model is only the last blank public tab, not an experiment.

| Model / reward | Machine / project | Submission | State at launch snapshot |
| --- | --- | --- | --- |
| DiffRhythm2 / all three, sequential | lab5090 | PID3884225 | Controller running; full30 baseline validation first |
| Muse / Richness | Gadi / wa66 |181297781 | PBS queued |
| Muse / Instrument | Gadi / iv96 |181297782 | PBS queued |
| Muse / Arrangement | Gadi / wa66 |181297783 | PBS queued |
| HeartMuLa / Richness | Gadi / wa66 |181297497 | PBS queued; new-setting acceptance gate then formal |
| HeartMuLa / Instrument | Gadi / wa66 |181297498 | PBS queued; acceptance gate then formal |
| HeartMuLa / Arrangement | Gadi / iv96 |181297503 | PBS queued; acceptance gate then formal |
| ACE-Step1.5 | Not submitted | n/a | Existing local assets found; backend integration not completed |
| SongBloom | Not submitted | n/a | Old isolated backbone smoke only, no full Miles worker acceptance |

The [one-shot launch receipt](launch-receipt.json) records scheduler states,
resource requests, source hashes and the DiffRhythm2 current phase. Queue
submission is not training completion. No overnight ETA is asserted while
Gadi has no available Hopper GPUs. No scheduled task or monitoring automation
was created; the local runner is a finite, user-authorized experiment queue.
No unrelated jobs were stopped or reprioritized.

## DiffRhythm2: Full Miles Chain Accepted

[Actual Ray/FSDP2 receipt](diffrhythm2-miles-acceptance.json), separate from
[the older isolated-kernel smoke](../framework-smoke/diffrhythm2-flow-receipt.json):

- Real `FSDPTrainRayActor`, native model backend and native Ray audio worker.
-112 attention/MLP modules,224 adapter tensors;6,029,312 trainable parameters.
- Two real audio reward groups and two optimizer updates, LR3e-4 both times.
- First identity KL0; second conditional KL approximately2.75e-5/dimension.
- Sampled versus teacher-recomputed log-prob mean gaps approximately2.51e-4
  and2.26e-4; both are logged, not silently replaced with detached new logps.
- Every scored audio SHA matches its generated PCM24-FLAC.
- Actor-to-worker LoRA transfer verifies a compute-FP16 adapter hash.
- A new actor reloads the saved adapter, optimizer moments, LR scheduler,
  Torch RNG and global/data cursor at2. Before/after receipts match.

This acceptance uses4s samples, not formal quality or throughput evidence.
Formal training requests270 updates, native60s generation, and all30 validation
conditions at0/50/100/270. No partial mean is a final validation result.
The currently accepted native weight-sync path supports **one FSDP rank only**;
it makes no distributed scaling claim.

Pinned upstream Miles revision:
`66f61901873a51be1de2072998090ac754fb18f3`.
The [project patch](miles-native-worker.patch) adds a native updater boundary,
defers unused SGLang/SP imports, and supports the public one-rank DeviceMesh API
on Torch2.9. It is not described as unmodified upstream. Ray2.53.0 and
PEFT0.14.0 are isolated overlays for native Transformers4.47.1; shared model
environments and CUDA installations were not upgraded.

The [frontend audit](diffrhythm2-frontend.json) verifies all300 conditions.
The original entry routes only English/Chinese although its native cleaner
also supports Korean. Three Korean lines are explicitly routed through that
native cleaner; punctuation separators are recorded. Unsupported alphabets
still fail, rather than dropping a condition. No split membership changed.

## Muse: Input Failure Repaired, New Formal Jobs

Old jobs181226207 /181226209 failed before training. The frozen train file
contains270 physical JSONL records but285 fragments under Python `splitlines()`:
valid Unicode lyric separators were mistaken for record boundaries.
[The strict parser](formal_inputs.py) reads physical lines, verifies both frozen
SHA256 hashes and270/30 counts, and fails on membership changes. The same bug
was corrected in the independent Muse validator.

The new PBS jobs use ms-swift/TRL + vLLM, all-linear backbone LoRA, explicit
rank8/alpha16, constant3e-4 and beta0.05. They retain the native512-token
training budget and fixed-seed MuCodec decoding; these are not claimed to be
60s training samples. All30 validation conditions are evaluated at0/270.
Long listening replay is a separate stage and is not yet recovered here.

Both Gadi projects were over scratch inode quotas, not disk byte capacity.
Only regenerable `.pyc` files in our experiment virtualenvs were removed.
Both quotas are now under allocation; new jobs disable bytecode writes.
Models, datasets, checkpoints and other projects' files were not removed.
Superseded submissions are listed in the receipt, not counted as active jobs.

## HeartMuLa: Gated Formal Submission

[The new TRL runner](train_heart_trl.py) derives from the frozen accepted runner
without editing that historical receipt. It uses all270 conditions, all eight
codebooks, global/local attention+MLP LoRA, constant3e-4 and beta0.05.
Each submitted job first checks two updates with these exact hyperparameters,
identity/drift KL and both global/local nonzero gradients. A successful gate
starts a fresh formal policy; its two acceptance steps are **not** counted
towards formal270. A failure stops that arm instead of claiming success.

Formal mode requests60s native rollouts on H200, activation checkpointing,
TRL optimizer/RNG checkpoints and all30 fixed validation inputs at0/270.
Long-sequence memory and full-run stability are not yet measured. Existing
head-only audio remains a labeled deprecated ablation, never substituted for
new backbone output.

## Manual Recovery

`collect_launch_receipts.py` is a one-shot read-only remote status collector.
It is not registered with any scheduler. Preserve each run's experiment JSON,
steps, checkpoints, full validation receipts and score-to-audio binding.
Only after complete receipts are recovered should new curves and listening
pairs replace the marked legacy outputs. Do not infer perceptual improvement
from the optimized reward alone.
