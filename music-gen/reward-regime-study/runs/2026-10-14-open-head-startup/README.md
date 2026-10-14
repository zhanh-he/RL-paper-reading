# Authorized Open-Model RL Launch

2026-10-14 AWST. Actual job submission and bounded startup verification, not
final results. No Codex schedules or overnight monitoring.

## Jobs

| Model | Richness | Instrument | Arrangement | Verified state |
| --- | --- | --- | --- | --- |
| DiffRhythm2 | lab5090 | same finite queue | same finite queue | Richness first5 real updates verified; remaining arms serial; queue PID3860601 |
| HeartMuLa | 181262219 | 181262220 | 181262221 | Gadi iv96, gpuhopper, priority1023, queued |
| SongBloom | 181262549 | 181262550 | 181262551 | Gadi wa66, gpuhopper, priority1023, queued |

Gadi job IDs have suffix `.gadi-pbs`. Preparation181261941/181261942 finished
successfully; both model assets and all300 native reference files are available.
Each GPU job requests one GPU,12CPU,160GB,12h, and runs270 real optimizer steps,
all30 validation conditions and five baseline/terminal replay examples. Submitted
is not running; startup compatibility for the two queued native generators is
not yet GPU-verified. Muse remains in its separately recorded older three jobs.

YuE2's three arms were already complete. Additional five-case long inference at
0/50/100/270 is queued as181262106 on Gadi wa66, priority1023. It is explicitly
`--skip-scoring`: new listening outputs are not a substitute for full validation;
score their immutable originals during manual recovery before publishing numbers.

## Trainable Scope

All three new models use rank8/alpha16 output-head LoRA, AdamW3e-4, G2,
beta0.01, seed42, native60-second budget with natural early termination.
Frozen CMI270/30 membership and all192 sealed WSB conditions are unchanged.

- HeartMuLa: codebook0 output head only. Frozen global/local/codec.
  Full-support FP32 categorical CFG1.5/temp1 for codebook0; native fixed local
  residual-codebook sampling is an environment, not a missing trainable likelihood.
- DiffRhythm2: DiT final proj_out only. Frozen transformer, MuQ, frontend and
  decoder. Official Flow-GRPO SDE, noise.1,31 transitions/block, CFGscale3.
  Full-block Gaussian log-prob sum; exact conditional KL per latent dimension.
  Comparison baseline must use the same SDE, not the original ODE. ODE deployment
  transfer is not yet evaluated. Float16 native DiT/decoder is fixed in both arms.
- SongBloom: final sketch classifier only. Frozen AR/flow/VAE/conditioners.
  Full-support categorical CFG1.5/temp1. It is not acoustic Flow-GRPO.
  A strict loader removes only365 duplicate VAE aliases proved tensor-identical;
  any other missing/unexpected key is fatal.

These are limited-capacity policy experiments, not full-backbone RL. Architecture,
input modality, trainable location, rollout budget and KL unit differ from the
existing YuE2/Muse experiments. A negative result cannot establish that an entire
model cannot improve with RL. No audible-improvement or formal-MOS claim is made.

One update/group means ratio clipping has no trust-region effect at that first
gradient. Frozen-reference regularization is explicit. Infrastructure/decoder
errors stop an arm rather than silently becoming low rewards or filtered rows.

## Provenance

| Source | Revision |
| --- | --- |
| DiffRhythm2 source | 7804f821b797b4f276090e1a9dcd37e97d9915d5 |
| DiffRhythm2 HF | 9aa15742e4889c0eb2e198db6fdab1facf1b6761 |
| HeartLib | 5858ca8f1ffe58d62be7c007ea55f1033c5ca456 |
| HeartMuLa happy-new-year3B | 41f6fc68490e11dc43fdabaa6b5767946408c903 |
| HeartCodec20260123 | f889dab0532cfa4bf459f2a3367eb6d346b8eeda |
| SongBloom community source | 990987ec94c49056a8dbf838987af93a37a1250d |
| SongBloom community HF base150s | 093351709a0efa263569cc4b679429d38f4508ec |
| Official Flow-GRPO helper | 879042cf5707f8b90daa98d147d7deac2317c5da |

SongBloom is an academic-use community restoration, not independently matched
to the unavailable author endpoint or the team's listening checkpoint. Model
asset receipts privately verify each downloaded LFS SHA. No weights are public.

## Evidence And Recovery

`experiment.json`, `steps.jsonl`, `progress.json` and three adjacent Python files
are the exact first DiffRhythm2 deployment snapshot. Source SHA values match
that archived deployment. Root-level trainers subsequently gained shared-baseline
cache checks and MP3 export for queued Gadi models; they are not falsely assigned
to the already-running DiffRhythm2 arm. Resume the latter from its archived source,
not by bypassing source-identity checks. Its initial source is also saved privately.

First-step old/current log-prob difference=0, reference KL=0, finite nonzero
gradient. Five checkpoint hashes are checked. This is a startup test on actual
formal training rows, not a mock result or a five-row benchmark. No validation
mean is published until30/30 receipts exist. CPU tests also verify categorical
reference identity, frozen weights, nonzero adapter gradient and shared baseline
byte/score identity across reward arms.

Private roots: lab5090 `/storage/zhanh_storage/open-music-rl-20261014`;
Gadi `/scratch/iv96/hm6416/open-music-rl-20261014`. Archive checkpoints, journals,
full validation receipts and replay receipts at manual recovery. Recheck completion
markers and actual process/PBS outcome; do not infer success from submitted IDs.
Lossless originals and SHA-bound scores remain private. Only192kbps48k MP3
listening copies may enter the repository; keep original/MP3 hashes separate.

At roughly40-46 seconds/update, the first DiffRhythm2 arm's training alone is
around3-3.5h, before validation/replays. This is a throughput extrapolation from
startup, not a guarantee for later lyric lengths, the full serial sweep, or Gadi
queue time. All-model completion by tomorrow morning is not promised.
