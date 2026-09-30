# YuE2 fixed-reference KL audit

`kl.json` measures mean per-token `D_KL(reference || checkpoint)` on three
fixed, reference-generated held-out semantic trajectories. The reference is
the shared one-step source LoRA. The distribution is normalized over codec
tokens and the permitted end token, before top-k, top-p, temperature, and
repetition-penalty sampling. The first 200 positions prohibit the end token.
This is **offline conditional KL**, not logged training-time or on-policy KL.
The exact 600-token contexts and seeds are in `reference_tokens.json`.

| LR | Step 5 | Step 25 | Step 50 | Step 100 |
| ---: | ---: | ---: | ---: | ---: |
| 2e-5 | 0.000790 | 0.000791 | 0.000797 | 0.000791 |
| 1e-4 | 0.000801 | 0.000799 | 0.000826 | 0.000874 |
| 1e-3 | 0.001161 | 0.009952 | 0.118716 | 0.497705 |
| 1e-2 | 0.387077 | 7.075164 | 12.792231 | 21.022778 |

Before accepting these numbers, the probe reloaded the identical reference
adapter through the same path as a candidate and obtained exactly zero KL on
all three contexts. The first probe had invalid `null` values from
`0 * (-inf - -inf)` at a forbidden end token. A subsequent version compared
adapters through different loading paths and gave a spurious nonzero
same-weight KL. Both versions were rejected; only `kl.json` with
`identity_kl: [0, 0, 0]` is reported here.

The reference's average probability mass inside the allowed codec vocabulary
was 0.99990-0.99996 across contexts. Thus full-vocabulary policy-loss
normalization is mathematically mismatched to sampling but its measured
probability-mass difference is tiny on these contexts. The 2e-5 and 1e-4
policies barely diverged by this measure, while the unregularized high-LR
policies drifted sharply and lost held-out reward. That supports high-LR
instability, not a claim that missing KL alone caused the low-LR reward
stagnation. These are only three short, capped contexts in BF16 inference;
broader on-policy KL and human listening remain necessary.
