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

These are pipeline checks, not adequately powered music-quality experiments.
Longer-run GRPO needs multiple prompts, independent held-out rewards, blind
listening, duration/loudness controls and explicit KL monitoring.
