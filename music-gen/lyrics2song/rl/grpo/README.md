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
fell from 3.0781 to 2.5313 on MuseCritic's own scale. A 100-prompt Gadi run
also completed 25 and 50 steps, with [paired 25-step](runs/2026-09-30-musecritic-longrun/step_000025/receipt.json)
and [50-step](runs/2026-09-30-musecritic-longrun/step_000050/receipt.json)
held-out replays. MuCodec decoding is stochastic, so those one-song audio
differences cannot be attributed solely to the adapter. `gadi_musecritic_smoke.pbs`
is the job template. Do not compare SongEval and MuseCritic numbers as if
they were the same reward.

These are still small pilot experiments, not adequately powered music-quality
claims. Longer-run GRPO needs blind listening, duration/loudness controls,
multiple seeds and explicit KL monitoring alongside held-out rewards.

## YuE2 data and objective audit

`yue2_songeval_longrun.py` contains the complete training and evaluation
requests: eight hand-written English style/two-line-lyrics prompts for training
and three separate hand-written prompts for held-out checks. No audio or
prompts from SongEval, WildSongBench, or CMI-RewardBench are training examples.
SongEval supplies the audio reward only. New runs write `experiment.json` with
the exact prompt lists, source adapter, reward backend, learning rate, and
semantic-token cap. The eight prompts cycle every eight updates, so data
diversity is a plausible limitation, not an established sole cause of reward
stagnation.

The current custom PyTorch update takes two on-policy rollouts per prompt,
normalizes their scalar rewards within that pair, and makes one optimizer
update on the same rollouts. Its `old` and `current` action log probabilities
are computed by the same policy before the update, making the ratio one at
gradient evaluation; the nominal PPO clip therefore provides no effective
trust region in this implementation. There is no reference-model KL penalty
or explicit duration, clipping, or lyric-fidelity guardrail. PyTorch supports
these constraints; their absence is a property of this training script, not
of the framework. Sampling also masks to codec tokens while the current
policy loss normalizes over the full vocabulary. The offline KL probe measures
codec-constrained fixed-reference divergence and allowed-vocabulary mass to
test whether that mismatch matters. This probe is **not** training-time KL.
With a group of two, non-tied standardized rewards always give advantages
`+1` and `-1`, regardless of the reward gap. In the 2e-5 run, 28 of the 99
updates from step 2 through 100 had an absolute pair gap below 0.1 yet still
received full-strength advantages. This could amplify near-ties or scorer
noise; it has not been isolated as the cause of stagnation.

The [high-learning-rate stress test](runs/2026-09-30-yue2-high-lr/README.md)
completed step 100 at both 1e-3 and 1e-2 on the 5090. Both finished below
their shared step-1 source LoRA on three fixed held-out prompts. These are
instability probes, not recommended settings. The first KL probe incorrectly
multiplied zero reference probability by an undefined `-inf - -inf` difference
for the forbidden end token; its `null` KL values are invalid. The corrected
`probe_yue2_reference_kl.py` masks that token and is queued to rerun after
`run_yue2_post_sweep.sh` finishes. MuseCritic's two-step smoke test passed;
its YuE2 run is now continuing toward step 100.

The [Muse SongEval prompt-matched replay](runs/2026-09-30-muse-matched/README.md)
regenerates the one-step adapter's held-out 0/1 audio using the same style,
lyrics and seed as the YuE2 comparison. On this one song, official SongEval
mean changed 2.8776 to 3.1095, while peak and RMS also rose. This is a
single-sample observation, not evidence of general quality improvement.

The [shared Muse replay receipt](runs/2026-09-30-muse-shared-replay/receipt.json)
uses one frozen baseline waveform for both SongEval and MuseCritic arms, with
the same prompt, seed, 500-token generation and 20-step MuCodec decoding.
Actual adapter replays exist at SongEval step 1 and MuseCritic steps 1, 25 and
50. Each audio file has both reward scores and a SHA-256 binding in its arm
receipt. The [demo](https://zhanh-he.github.io/RL-paper-reading/demos/#lyrics)
shows missing stages as unavailable, not as interpolated results. MuseCritic
step 100 is queued as Gadi job `180204792.gadi-pbs`; dependent step 300 is
`180213393.gadi-pbs`. The latter resolves the former's complete optimizer
checkpoint at runtime through `gadi_musecritic_resume_chain.pbs`. SongEval
steps 25/50/100/300 are not yet trained. These are one-song, one-seed
comparisons with separate reward scales, not evidence of preference gains.
