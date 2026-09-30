# Muse SongEval, YuE2-matched held-out replay

This is a fresh held-out 0/1-step replay of the existing one-step Muse
SongEval-GRPO adapter. It does not add an optimizer update. The Muse prompt
template retains its model-specific wrapper, but its style fields, lyrics,
seed 5101 and 500-token forced length are fixed across A/B and match the
published YuE2 style/lyrics/seed. Audio uses released Muse-0.6b and MuCodec
weights; SongEval is scored with its official five-dimension evaluator.

| Held-out audio | SongEval mean | Peak | RMS | Full-scale samples |
| --- | ---: | ---: | ---: | ---: |
| Frozen Muse | 2.8776 | 0.475 | 0.054 | 0% |
| One SongEval-GRPO update | 3.1095 | 0.972 | 0.094 | 0% |

The single sample's higher SongEval score co-occurs with greater loudness.
MuCodec decoding is stochastic; one A/B replay cannot establish a causal
quality gain or generalization. Training used two rollouts and one optimizer
step. [Receipt](receipt.json), [replay code](../../muse_matched_heldout.py),
and the [public A/B page](https://zhanh-he.github.io/RL-paper-reading/demos/#lyrics)
allow the claim to be checked without publishing weights.
