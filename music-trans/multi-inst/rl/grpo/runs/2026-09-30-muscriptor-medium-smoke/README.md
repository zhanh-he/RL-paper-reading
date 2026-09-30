# MuScriptor-medium GRPO feasibility, 30 September 2026

This run answers whether a real MuScriptor GRPO update can be executed on the
available research hardware. It does **not** make a presentation-grade
transcription claim. No weights or source dataset audio/MIDI are published.

## Setup

- Authorized MuScriptor-medium offline checkpoint from the team's earlier
  benchmark cache; SHA-256
  `ac80adbdf85d87231735fd948af7013441c0afced316c4e9067fd5d8a7fb97ec`.
- Freeze the acoustic conditioner and decoder transformer. Update the final
  token projection only. Four sampled MIDI token rollouts per group, 384-token
  cap, 50 requested steps, 50 effective optimizer updates, two clipped policy
  epochs per group, AdamW `1e-5`, KL coefficient `0.01` to the fixed initial
  token head. The sampler and replay both mask the two non-vocabulary IDs.
- Pitch-only reward on one 5 s YouChorale training excerpt: `0.45` onset F1,
  `0.25` complete-note F1, `0.30` frame F1. Onset tolerance is 50 ms;
  complete-note also requires offset within max(50 ms, 20% reference length).
- Deterministic greedy decoding for before/after comparison. Source song was
  chosen from an existing benchmark where MuScriptor had a measurable
  baseline. It is *not* an unbiased training-set sample.

## Measured result

| Scope | Stage | Frame F1 | Onset F1 | Complete-note F1 |
| --- | --- | ---: | ---: | ---: |
| Training song, first 5 s | Frozen | 0.482 | 0.179 | 0.123 |
| Training song, first 5 s | 50 steps | 0.490 | 0.292 | 0.213 |
| 8 disjoint songs, first 5 s macro | Frozen | 0.633 | 0.043 | 0.037 |
| 8 disjoint songs, first 5 s macro | 50 steps | 0.637 | 0.109 | 0.103 |

The [training receipt](receipt.json) records the four rollout rewards and
variance for every step, plus KL against the fixed reference. A direct
comparison of the saved token head with the original checkpoint found a
maximum absolute parameter change of `0.000826` (mean `0.000121`), confirming
that this is not an inference-only replay. The [disjoint audit](disjoint8.json) gives anonymous
paired scores, without song IDs. These eight songs were chosen by a
deterministic sort rule excluding the training song, but the audit was
designed after this feasibility run. Onset and complete-note F1 improved on
3 of 8 songs and were unchanged on 5; frame F1 declined on 2 songs. One
training song, one seed, five-second
excerpts and no instrument/SATB track evaluation cannot justify a broad
generalization or anti-reward-hacking claim. Treat this as a promising
engineering smoke and prioritize the ChoralStream held-out ablation for the
presentation.

[Training code](../../muscriptor_grpo_smoke.py) ·
[paired evaluation code](../../evaluate_muscriptor_head.py)
