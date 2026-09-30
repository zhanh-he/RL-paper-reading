# YuE2 SongEval reward and training protocol

This is the actual 2026-09-30 academic pilot, not a recipe for the private
Huawei model. The released YuE2-3B semantic generator and YuE2-VAE are used
on lab5090. The public [training implementation](../rl/grpo/yue2_songeval_longrun.py)
and [run receipts](../rl/grpo/runs/2026-09-30-yue2-longrun/README.md)
define the measurements.

## Reward

For one decoded audio clip, the [official SongEval](https://github.com/ASLP-lab/SongEval)
model returns `Coherence`, `Musicality`, `Memorability`, `Clarity`, and
`Naturalness`. The scalar optimized here is their unweighted arithmetic mean:

`R(audio) = (Coherence + Musicality + Memorability + Clarity + Naturalness) / 5`.

There is no explicit term for correct lyrics, prompt adherence, vocal
intelligibility, duration, silence, repetition, peak, clipping, or signal
quality beyond what SongEval may implicitly reflect. It is an audio-only
aesthetic proxy, not a reference-song score or human preference measurement.
The official inference path first decodes audio as mono at 24 kHz; output
energy near 22.05 kHz therefore does not show what the reward model heard.

## Optimization

- Eight original two-line English verse prompts are training inputs. Each
  update takes one prompt and samples two fresh on-policy songs.
- The two scalar rewards become group-relative advantages:
  `A_i = (R_i - mean(R_1,R_2)) / max(std(R_1,R_2), 1e-8)`.
  With two unequal rewards, these are essentially +1 and -1. The absolute
  reward difference does not scale the gradient.
- The code computes token log probabilities of the generated semantic tokens
  under the current LoRA policy, then applies one AdamW update at `2e-5`.
  LoRA changes only `q_proj` and `v_proj` on the semantic generator.
- The nominal probability-ratio clip is `[0.8, 1.2]`, but old and current
  probabilities come from the same policy immediately before this *single*
  gradient step. The ratio is 1 at gradient evaluation, so PPO clipping is
  not an active safeguard in this implementation. There is no explicit
  reference-policy KL term. Calling this a GRPO-style group-relative policy
  gradient pilot is more precise than claiming full multi-epoch GRPO.
- Each song is capped at 600 semantic tokens. Held-out receipts record
  whether this truncates the song; truncated song quality is not full-song
  quality. Optimizer updates, not generated samples, are counted as steps.
- The 0-step frozen model and 1-step source LoRA were evaluated in the same
  inference configuration. Updates 2 onward resume from that source LoRA;
  AdamW state was newly initialized at update 1 and is saved in subsequent
  checkpoints. Total target: 300 updates, not 300 extra updates.

## Evaluation and guardrails

Three disjoint original prompts are evaluated with seeds 5101-5103 at each
reported milestone. The score in the summary table is the mean of three
per-song five-dimensional means; the single A/B audio on the website is only
prompt 0. Also record peak, RMS, fraction of samples at or above 0.999, and
semantic truncation. The fixed prompt/seed pair controls input and sampling,
but three prompts and one training seed do not establish generalization.

Before claiming that SongEval optimization improved music, conduct blinded
same-loudness A/B listening, lyric transcription/alignment checks, and
multiple train/eval seeds. The 47-clip [waveform perturbation audit](audits/2026-09-29/README.md)
found a gain effect; RMS-matched clipping was not consistently rewarded. It
cannot diagnose the private Huawei SongEval-GRPO observation without its
outputs or training logs.

MuseCritic is a separate five-aspect reward model and its scores must not be
compared numerically with SongEval. CMI-RewardBench is a benchmark for reward
models, not itself a callable training reward; a CMI reward model would be a
separate training arm, not part of this run.
