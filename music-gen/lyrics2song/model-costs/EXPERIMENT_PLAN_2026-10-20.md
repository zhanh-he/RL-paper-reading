# ACE-Step1.5 / DiffRhythm2: Timing Audit and Experiment Plan

Status: **proposed, not launched**. Updated 2026-10-20 AWST.

[Online plan](https://zhanh-he.github.io/RL-paper-reading/demos/#lyrics-model-plan) | [Model Cost](https://zhanh-he.github.io/RL-paper-reading/demos/#lyrics-model-costs) | [Machine-readable plan](experiment-plan.json) | [Pinned source audit](inference-boundaries.json)

## What We Can Actually Conclude

The former ACE 60s/300s values, 0.5s/2.5s, were arithmetic extrapolations from the project-page 240s/2s claim. They were **not a local measurement**, did not select a planner size, and did not bind a DiT checkpoint or timing boundary. They have been withdrawn from the table. We cannot establish whether the claim includes planning, rendering and decoding, nor relabel it as time to first audio (TTFA).

Our exact parameter audit is **v15-base**, not Turbo or XL: generator 2,393,872,518; caption encoder 595,776,512; VAE 168,695,426. The separately audited optional named 1.7B planner is 1,854,243,840. The complete loaded component count with that planner is 5,012,588,296. The caption encoder and optional planner are different models. We have no complete local ACE60/300 timing receipt and no exact Turbo/XL parameter audit in this table.

Diff's approximately23.86s is the median of29 complete60s baseline worker receipts. It includes offload/transfers, conditioning/cache, decoding, saving and cleanup. The300s figure, approximately119.3s, is **only a linear extrapolation**. Individual overheads were not timed, so no numerical causal decomposition is defensible yet.

## Why Smaller Does Not Necessarily Mean Faster

ACE's distilled Turbo does a small number of whole-latent refinement steps. Its optional planner can use **nano-vLLM**; this does not accelerate the DiT or VAE automatically. Base and Turbo must not share a speed label. Planner-off is a different pipeline, not a free optimization of planner-on generation.

Diff is block flow matching. In the audited configuration, 5Hz latent and10 latent frames/block yield approximately2s blocks. A60s request has about30 sequential blocks. With32 Euler time points and conditional/unconditional CFG calls, the sampling path entails about30 x31 x2 =1,860 short-block calls, plus cache updates. This is **not a FLOPs ratio** versus8 whole-sequence ACE passes. Diff already has block KV caching and SDPA. The pinned loader disables FlexAttention; we have not verified compile/graph usage or profiled the actual attention kernel. Thus we know there are different execution paths and measurement boundaries, not that one particular optimization explains a measured speedup.

Primary evidence: [ACE profiler](https://github.com/ace-step/ACE-Step-1.5/blob/ca1e85fe9430179831e6bc6be790c332190a3866/profile_inference.py), [ACE planner backend](https://github.com/ace-step/ACE-Step-1.5/blob/ca1e85fe9430179831e6bc6be790c332190a3866/acestep/llm_inference.py), [ACE report](https://arxiv.org/html/2602.00744v2), [Diff block sampler](https://github.com/ASLP-lab/DiffRhythm2/blob/7804f821b797b4f276090e1a9dcd37e97d9915d5/diffrhythm2/cfm.py), [Diff loader](https://github.com/ASLP-lab/DiffRhythm2/blob/7804f821b797b4f276090e1a9dcd37e97d9915d5/inference.py#L84). Existing local receipts remain linked in [README](README.md).

## Fixed Configuration Matrix

| Profile | Planner | Generator | Role |
| --- | --- | --- | --- |
| ACE Base | off; caption encoder remains | 2B-class Base,50 steps, native CFG recorded | Existing parameter audit's latency baseline |
| ACE Turbo off | off; caption encoder remains | 2B-class Turbo,8 steps, no CFG | Isolate planner cost |
| ACE Turbo PyTorch | same1.7B, thinking/audio codes on | Same Turbo | Paired planner backend baseline |
| ACE Turbo nano-vLLM | same1.7B, identical generation settings | Same Turbo | Primary complete ACE pipeline |
| Diff worker | native style conditioning | Existing1.136B,32 time points, CFG2, FP16 | Reproduce worker boundary |
| Diff resident | identical conditioning/weights | Same sampler/cache/precision; GPU resident | Primary complete Diff pipeline |

0.6B/4B planner and4B XL are **out of the first sweep**. Turbo's precise parameters must be counted from its actual checkpoint before publication. Record actual runtime switches and resolved configuration rather than trusting requested flags. Unsupported combinations remain failures, not silently substituted variants.

## Measurement Contract

- Batch1, same dedicated GPU for paired comparisons. Use an idle5090 or independent interactive allocation; never evict existing jobs. If capacity requires A100-80GB, move the whole paired group. No cross-hardware backend speedup ratio.
- Freeze a manifest before launch: five existing CMI validation conditions, three with explicit instrumentation constraints and two without; exact IDs, input texts, lyrics, pre-fixed60s/300s versions and SHA256. Use seeds5101/5102/5103. Native style/audio requirements are recorded separately; common text/lyrics do not imply identical modalities. WSB remains sealed.
- Main latency starts with a loaded model receiving inputs and ends at a complete CPU waveform, with CUDA synchronization at the outside boundaries. Include tokenization, input encoding, optional planner, sampler, decoder and necessary transfers. Record actual output duration and RTF=wall/audio.
- Separately time playable MP3 delivery, cold load, compile, warmup and genuine TTFA. An HTTP task ID, first token or request acknowledgment is not first audio. Non-streaming profiles have TTFA n/a, not an invented streaming measurement.
- Do not add per-stage synchronization to the principal latency pass. Use a separate profiling pass for component times/NFE/block counts. Report GPU peak allocated/reserved and process memory, plus hardware/runtime versions.
- Public audio is MP3 only,192kbps/48kHz, no loudness normalization; retain separate original/MP3 hashes. Private lossless scoring originals stay off the repo. Reward scoring is outside latency.
- Early endings, OOM, timeouts and unsupported300s are visible failures/actual durations, never padding, looping or invented extrapolations. Report planned/complete N and failures. Median/p90 across15 condition-seed requests per main profile/length is a small-sample descriptive summary, not a precise production tail guarantee.

## Staged Budget

1. First run one condition/seed, six profiles,60s before300s. These12 requests count toward the full matrix if they satisfy the identical contract; no repeat for appearances. Separate short warmups and diagnostic profiles are recorded outside the request count.
2. Main profiles (Turbo1.7B nano-vLLM and Diff resident):5 conditions x3 seeds x2 lengths x2 profiles =60 complete-generation requests. Four other profiles:1 of those conditions x3 seeds x2 lengths =24 requests. **84 measured requests total**, not84 independent songs or a full-dataset quality evaluation.
3. Proposed first allocation cap:4 GPU-hours, not a completion-time prediction. At the cap, return the actual N/status. Optional Diff compile/graphs/FlexAttention is a later separately versioned optimization only after numerical/compatibility checks. Do not assume such switches are supported or interchangeable.
4. Use measured rollout, reward, old/reference log-prob and update costs to budget future270-step RL: initially G=2; G=4 only a cost scenario until measured. Training throughput is not inference latency. Preserve approved LR/beta and split; any new RL submission or change of trainable policy needs explicit follow-up approval.

## Decisions After Measurement

Keep the fastest **accepted complete pipeline**, not the fastest isolated timer. Retain paired SongEval/SongBench assessment and listening for quality/cost trade-offs, without calling automated scores MOS. Do not require identical diffusion step counts as a proxy for equal quality. Distilled Turbo versus Base is a model comparison; changing the sampler or planner can change quality.

The inference engine does not supply a valid GRPO policy automatically. Before a new RL run, verify the trainable modules, old/new sampling distribution, fixed-reference KL and whether the chosen diffusion path admits the intended FlowGRPO likelihood. Neither the website audit nor this plan starts/stops training, requests interactive GPUs or creates a schedule.
