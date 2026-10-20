# Approved Priority Launch

[Plan](../../SONGBENCH_MEAN7_PRIORITY_PLAN_2026-10-20.md) |
[One-shot evidence](launch.json) |
[Online execution / historical results](https://zhanh-he.github.io/RL-paper-reading/demos/#lyrics-model-plan)

This record is a manually collected launch snapshot, not a completed result or an automatic monitor.

## Original Muse First

- Muse SP Richness:5090 formal270-step job, PID3982658.
- Muse SongBench-Instrument: Gadi wa66/Hopper,181718185.gadi-pbs.
- Muse SongBench-Arrangement: Gadi iv96/A100,181718187.gadi-pbs.

Each starts from the exact original non-GRPO Muse release; full weight/config identities are in the evidence. No old GRPO adapter or merged checkpoint is loaded. Rank8/alpha16 all-linear backbone LoRA has5,046,272 trainables; LR3e-4 constant, beta0.05, G2, seed42. The target remains the historical SP/subhead **minus guard**, not pure Mean7. All270 training/30 validation conditions are frozen. These controls retain512 native completion tokens and fixed-seed10.24s MuCodec decoding, not the new60s Mean7 protocol. Baseline and final validation must both complete30/30 before reporting a mean as complete. Across-host baselines need a canonical recovery check before public A/B replay.

The separate two-update acceptance completed with nonzero gradients, KL0 then0.000505387, four scored-audio hashes verified, and hashed adapter/optimizer/RNG checkpoint files. It is not a training result, quality claim or cold-restore acceptance. Empty request-ID fields from the native reward callback are not used as identifiers; audio SHA binds the score to the decoded bytes.

## Mean7 Successors

DiffRhythm2 pureMean7: finite dependency process3983215 on5090 waits for Muse-SP training **and full validation** to complete. Muse failure blocks the successor. The native Miles two-step/cold-actor-restore gate runs before fresh60s-budget270-step training. The scorer uses the seven-head raw arithmetic mean, no SP separator or extra penalty. This is a queued successor, not already running Mean7 training.

YuE2 pureMean7: Gadi wa66/Hopper181718819.gadi-pbs. Native TRL imports and tokenizer checks passed on5090 and Gadi; GPU acceptance is pending. The submitted job gates two native60s-budget updates, separate-process checkpoint2→3 continuation, then fresh270 formal. Any gate failure stops; no silent fallback. This continuation gate does not establish bit-exact resumed/uninterrupted equivalence. MuseMean7 remains behind the three original-base controls. HeartMuLa, SongBloom and ACE are deferred.

## Scope and Reproduction

The manual collector is `backbone-rl/collect_priority_launch.py`. It reads existing logs/checkpoints and scheduler state without launching evaluation or inference. The snapshot retains all new source SHAs, original Muse identity, acceptance diagnostics, frozen split hashes, scheduler requests and progress at collection time.270 train/30 valid/121 CMI test and192 reserved external WSB remain unchanged. No test result is available for this launch.

Held superseded Muse submissions181714731/736/740 were deleted before their corrected replacements were submitted. No unrelated job was canceled. Training/scoring FLAC stays private; eventual public replay copies must be MP3 only, separately hashed, with no normalization. No recurring task or heartbeat was created.
