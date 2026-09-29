# Lyrics-to-song GRPO

The [YuE2 SongEval pilot](runs/2026-09-29-yue2/README.md) is an on-policy,
two-rollout, one-step LoRA update with a deterministic held-out before/after
audio pair. Its held-out SongEval mean fell; it did not produce clipping. See
the [controlled reward audit](../../rewards/audits/2026-09-29/README.md)
before making claims about reward hacking.

`yue2_songeval_pilot.py` and `verify_yue2_pair.py` reproduce that pilot with
local YuE2 weights and the official SongEval checkout. The separate Muse path
uses `muse_songeval_pilot.py` and `decode_muse_tokens.py`, with released
Muse-0.6b and MuCodec weights. Each run must record training group size,
reward values, parameter change, untrained replay determinism, held-out scores,
audio peaks and whether the song was capped or truncated.

The [YuE2 longer run](runs/2026-09-30-yue2-longrun/step_000100/receipt.json)
trained from the previous day's one-step LoRA, with 8 original training prompts,
2 rollouts per update and 3 independent fixed held-out prompts. Its measured
0/1/5/50/100-update held-out SongEval means are
3.8403/3.6030/3.6160/3.5704/3.6063; the [replay](https://zhanh-he.github.io/RL-paper-reading/demos/#lyrics)
includes one same-prompt/seed audio pair at each milestone. None establishes
improvement or full-scale clipping.

A separate [MuseCritic one-step online GRPO](runs/2026-09-30-musecritic-smoke/receipt.json)
completed on Gadi. Its [held-out pair](runs/2026-09-30-musecritic-heldout/receipt.json)
fell from 3.0781 to 2.5313 on MuseCritic's own scale. Another Gadi job is
training 50 steps from about 100 public MuseCritic prompts, with dependent
held-out evaluation at 25 and 50; those are not results yet. `gadi_musecritic_smoke.pbs`
is parameterized for that run. Do not compare SongEval and MuseCritic numbers
as if they were the same reward.

These are still small pilot experiments, not adequately powered music-quality
claims. Longer-run GRPO needs blind listening, duration/loudness controls,
multiple seeds and explicit KL monitoring alongside held-out rewards.
