# YuE2 SongEval high-learning-rate stress test

These two 5090 runs used the same YuE2 one-step source LoRA, eight original
hand-written two-line English lyric prompts (cycled during training), and
three distinct fixed-seed hand-written held-out prompts. They did not train
on SongEval, WildSongBench, or CMI-RewardBench examples. SongEval's five-score
mean was the only optimization reward. The exact prompt lists and settings are
in each arm's `experiment.json`; the train script is
`../../yue2_songeval_longrun.py`.

| Optimizer step | 1e-3 held-out mean | 1e-2 held-out mean |
| ---: | ---: | ---: |
| 0 (base model) | 3.8403 | 3.8403 |
| 1 (shared source LoRA) | 3.6030 | 3.6030 |
| 5 | 3.6504 | 4.0533 |
| 25 | 3.7188 | 1.5803 |
| 50 | 3.6722 | 1.5034 |
| 100 | 3.0909 | 1.7783 |

Each value is the mean of **three** held-out generations, not an estimate of
population music quality. All 36 held-out waveforms reached the 600-token
semantic cap (about 24 seconds), so full-song structure is untested. The 1e-3
arm trained to step 100 with a saved adapter and optimizer state; the score
fell by 0.5121 versus the step-1 LoRA. The 1e-2 arm also completed step 100,
but its apparent step-5 rise did not persist; its final score fell by 1.8247
versus step 1.

At 1e-2 step 25 all three clips reached peak 1.0, with 0.15-0.19% of samples
near full scale and RMS 0.379-0.393. At step 100 all three still touched 1.0,
but near-full-scale fractions were only 0.002-0.004%. The reward had already
collapsed; these data show unstable training and some clipped outputs, **not**
successful reward exploitation. The 1e-3 arm had a small near-full-scale
fraction (0.063%) on one step-25 song and none at step 100.

The method has no reference KL constraint, and its single-update ratio clip
does not bound the policy update. These stress tests cannot isolate whether
that omission, the eight-prompt data regime, scorer limitations, or another
factor caused the low-LR reward stagnation. The separate
[fixed-reference KL audit](../2026-09-30-yue2-kl-audit/README.md) found
large policy drift at these high learning rates, but should not be confused
with training-time KL.
