# ACE-Step 1.5

Upstream: [code and model links](https://github.com/ace-step/ACE-Step-1.5) and [official inference task guide](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/INFERENCE.md). For vocal completion, choose a checkpoint that actually supports `complete` (base or XL-base), not a faster SFT/turbo variant with a different task contract. The model is flow/diffusion-based; a token-level GRPO trainer is not supplied by this directory.

## Local inference receipt

On 2026-09-29, the base model loaded on lab5090's RTX 5090. `guide_vocal.py` made an original 16-second synthetic sung-vowel signal. `run_complete.py` used the official `AceStepHandler` with `task_type="complete"`, seed 29 and 50 diffusion steps. It generated a non-silent 16-second stereo result. [The public demo](https://zhanh-he.github.io/RL-paper-reading/demos/#vocal) contains the exact input and output WAVs, whose hashes are in [receipt.json](receipt.json).

This is **not GRPO** and not evidence of good accompaniment. The guide is synthetic, not a person singing. The `complete` output may be a vocal-plus-instruments mix, so do not calculate isolated accompaniment coverage or vocal preservation on it. A real follow-up needs authorized clean vocal recordings, identifiable accompaniment stems, blind listening, and same-input pre/post outputs.
