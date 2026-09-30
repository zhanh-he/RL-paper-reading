# YuE2 SongEval high-learning-rate stress arm

**Status:** 5-update held-out replay measured; 100-update target in progress.
This is a deliberately higher-learning-rate diagnostic, not a quality claim.

The run uses the [same implementation](../../yue2_songeval_longrun.py), source
one-step LoRA, 8 original training prompts, 2 on-policy rollouts per update,
3 original held-out prompts, seeds 5101-5103, and 600 semantic-token cap as
the [LR 2e-5 control](../2026-09-30-yue2-longrun/README.md). Only AdamW
learning rate changes to `1e-4` (5x). Frozen 0-update and shared 1-update
held-out FLAC files are byte-identical to the control; the public demo reuses
them. The [reward and optimizer protocol](../../../../rewards/songeval-grpo-protocol.md)
states why this single-update-per-group method has no active ratio clip or
reference KL safeguard.

| Optimizer update | 0 | 1 | 5 |
| --- | ---: | ---: | ---: |
| LR 1e-4: SongEval Mean5, 3 held-out prompts | 3.8403 | 3.6030 | 3.4285 |
| LR 2e-5 control, same prompts | 3.8403 | 3.6030 | 3.6160 |

At update 5, all three held-out clips hit the semantic token cap. The first
clip has 12 samples at or above 0.999 absolute amplitude, including 3 at
full scale; its longest consecutive run is 4 samples in one channel. The
other two clips have no such samples. This is a transient peak diagnostic,
not evidence of pervasive clipping or of reward hacking: the high-LR mean
fell below both baseline and the control's update-5 mean. The update-5 adapter SHA-256
is `78df842598b87ed7fb2c0c834241eb2122e6e25fed3ba4bf30a409df7dc52749`.
The [receipt](step_000005/receipt.json) includes five dimensions and signal
diagnostics for every held-out clip. The first clip is replayable in the
[demo](https://zhanh-he.github.io/RL-paper-reading/demos/#lyrics).

Two initial launches OOMed when another legitimate 5090 experiment grew its
GPU allocation; neither produced a public post-update-3 evaluation. The
current run resumes from its saved step-3 adapter and AdamW state after that
other process exited. A lower score at five steps is not reward hacking.
