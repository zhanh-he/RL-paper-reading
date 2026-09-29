# Lyrics to song

**Contract:** lyrics and style prompt in, a complete song or declared-length excerpt out. Evaluate lyric adherence, musical/production quality and long-range structure separately, under matched decoding length and seed policy.

- [Models](models/README.md) start with Muse and YuE2 as different-size systems, not a pure parameter-count causal comparison.
- [Rewards](rewards/README.md) specify independent SongEval, MuseCritic, CMI-RM and audio-aesthetic arms.
- [RL](rl/README.md) separates DPO from GRPO and requires a true base-versus-post checkpoint comparison.

CMI-RewardBench is a benchmark, while CMI-RM is a scorer. Neither a Best-of-K
filter nor a single reward-score increase is an online training result. We now
have two genuine but deliberately tiny on-policy SongEval GRPO pilots with
same-seed before/after audio: [YuE2](rl/grpo/runs/2026-09-29-yue2/README.md)
and [Muse](rl/grpo/runs/2026-09-29-muse/README.md). Both held-out scores fell
after one step; neither output clipped. The [47-clip SongEval perturbation
audit](rewards/audits/2026-09-29/README.md) finds a clean-loudness bias, not a
consistent same-RMS preference for clipping. These are engineering evidence,
not full-song quality or long-run GRPO conclusions.
