# Lyrics to song

**Contract:** lyrics and style prompt in, a complete song or declared-length excerpt out. Evaluate lyric adherence, musical/production quality and long-range structure separately, under matched decoding length and seed policy.

- [Models](models/README.md) start with Muse and YuE2 as different-size systems, not a pure parameter-count causal comparison.
- [Rewards](rewards/README.md) specify independent SongEval, MuseCritic, CMI-RM and audio-aesthetic arms.
- [RL](rl/README.md) separates DPO from GRPO and requires a true base-versus-post checkpoint comparison.
- [Public research map](research/README.md) surveys native model inputs,
  post-training feasibility, reward families and dataset protocols, with sources
  and matching Models / Rewards / Dataset Protocol panels in the online demo.

## Formal CMI-Pref Experiments

The [2026-10-12 recovery](rl/grpo/runs/2026-10-12-formal-recovery/README.md)
uses frozen 270 train / 30 validation / 121 sealed official-test conditions.
Both models receive their native text+lyrics projection, without reference
audio. YuE2-SongEval and YuE2-MuseCritic each have three audited 270-step
learning-rate arms. Earlier Muse small-LR arms remain archived; six replacement
arms use the matched 2e-5 / 1e-4 / 3e-4 grid. The live registry records actual
status, not fabricated final scores, and independent Muse inference is paused.

The [public demo](https://zhanh-he.github.io/RL-paper-reading/demos/#lyrics)
shows combined training reward beside sampled training KL, separate fixed
validation tables, and three paired long-inference conditions with one frozen
baseline per model when replay is available. Only MP3 listening copies are
published in the GitHub release; lossless scoring originals remain private,
with separate SHA256 bindings. Plots use fixed waveform bounds and mel scale.
Long replay is not short validation, and reward increases are not
listening-based quality claims.

## Earlier Engineering Pilots

CMI-RewardBench is a benchmark, while CMI-RM is a scorer. Neither a Best-of-K
filter nor a single reward-score increase is an online training result. Earlier
work included two genuine but deliberately tiny on-policy SongEval GRPO pilots with
same-seed before/after audio: [YuE2](rl/grpo/runs/2026-09-29-yue2/README.md)
and [Muse](rl/grpo/runs/2026-09-29-muse/README.md). Both held-out scores fell
after one step; neither output clipped. The [47-clip SongEval perturbation
audit](rewards/audits/2026-09-29/README.md) finds a clean-loudness bias, not a
consistent same-RMS preference for clipping. These are engineering evidence,
not full-song quality or long-run GRPO conclusions.
