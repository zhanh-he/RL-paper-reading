# YuE2 SongEval high-learning-rate stress arm

**Status:** 100 optimizer updates complete; 0/1/5/25/50/100 held-out stages
measured. The 100-update replay and VAE-clamping probes are verified.
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

| Optimizer update | 0 | 1 | 5 | 25 | 50 | 100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LR 1e-4: SongEval Mean5, 3 held-out prompts | 3.8403 | 3.6030 | 3.4285 | 3.6296 | 3.7639 | 3.4296 |
| LR 2e-5 control, same prompts | 3.8403 | 3.6030 | 3.6160 | 3.5629 | 3.5704 | 3.6063 |

Mean of each SongEval dimension across the same three held-out clips:

| Dimension | Frozen 0 | LR 1e-4 step 5 | LR 1e-4 step 25 | LR 1e-4 step 50 | LR 1e-4 step 100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Coherence | 3.9588 | 3.5941 | 3.7198 | 3.8633 | 3.5391 |
| Musicality | 3.8736 | 3.5269 | 3.6942 | 3.8024 | 3.5068 |
| Memorability | 3.9619 | 3.4649 | 3.6777 | 3.8488 | 3.4956 |
| Clarity | 3.7849 | 3.3228 | 3.5970 | 3.7023 | 3.4124 |
| Naturalness | 3.6226 | 3.2337 | 3.4595 | 3.6026 | 3.1942 |

All five remain below the frozen baseline at steps 50 and 100. These are model-judge
scores, not independent listening ratings.

| Held-out prompt | Frozen mean | High-LR step 5 | High-LR step 25 | High-LR step 50 | High-LR step 100 | Near-full-scale samples, step 5 / 25 / 50 / 100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (timeline A/B audio) | 3.7688 | 3.2961 | 3.7539 | 3.7880 | 3.3812 | 12 / 37 / 2 / 11 |
| 1 | 3.7869 | 3.6036 | 3.4847 | 4.0407 | 3.3145 | 0 / 0 / 0 / 0 |
| 2 | 3.9653 | 3.3857 | 3.6503 | 3.4630 | 3.5932 | 0 / 0 / 0 / 2 |

Each prompt's step-25 reward remains below its own frozen baseline. Audio
content and RMS also change, so the increasing near-full-scale count on
prompt 0 is an observation, not a controlled clipping effect.
At step 50, two prompts score above their own frozen scores while the third
falls by 0.5023; the three-song mean is still 0.0764 below baseline. All
three clips are again semantic-token truncated. Prompt 0 has only two
near-full-scale samples, neither exactly full scale.
At step 100, all three scores are below their own frozen baselines. The
three-song mean fell by 0.3343 from step 50 and by 0.4107 from baseline;
all three clips again hit the 600-token cap. This is late-stage degradation,
not an observed increase in the optimized reward.

At update 5, all three held-out clips hit the semantic token cap. The first
clip has 12 samples at or above 0.999 absolute amplitude, including 3 at
full scale; its longest consecutive run is 4 samples in one channel. The
other two clips have no such samples. This is a transient peak diagnostic,
not evidence of pervasive clipping or of reward hacking: the high-LR mean
fell below both baseline and the control's update-5 mean. The update-5 adapter
SHA-256 is `78df842598b87ed7fb2c0c834241eb2122e6e25fed3ba4bf30a409df7dc52749`.
At update 25, the first clip has 37 near-full-scale samples (2 at full scale),
with a longest consecutive run of 17 samples; the other two have zero.
The three-song mean remains below the frozen baseline, so this is still not
evidence that the reward prefers clipping. The update-25 adapter SHA-256 is
`657f08d65797c648f9c121ac514b08e6c9b7c392b8018f0ac3e36bec51ec6504`.
The [update-5](step_000005/receipt.json),
[update-25](step_000025/receipt.json), and
[update-50](step_000050/receipt.json), and
[update-100](step_000100/receipt.json) receipts include five dimensions and
signal diagnostics for every held-out clip. The first clip of each is replayable in the
[demo](https://zhanh-he.github.io/RL-paper-reading/demos/#lyrics).

The [checkpoint comparison](adapter-delta-step25.json) verifies that both
step-25 adapters contain the same 224 LoRA tensor keys. Relative to the shared
step-1 adapter, their concatenated float32 parameter-delta L2 norms are
`0.0985` (control) and `0.4937` (stress), a 5.01x ratio. This verifies a
larger policy update, not better music. The control step-25 held-out score
was measured after this parameter check and is included above. The stress
run's [steps.jsonl](steps.jsonl) contains exactly the contiguous optimizer
updates 2-100, with finite rewards and gradient norms throughout. The
[VAE-to-FLAC probe](../../probe_yue2_float_path.py) compares the VAE's
pre-clamp float output, YuE2 pipeline's `clamp(-1, 1)` float output, and PCM-24
FLAC. All five [probe receipts](probes/) match the published FLAC files by
SHA-256 and decoded samples. On held-out prompt 0, the high-LR VAE exceeded
unit amplitude at steps 5/25/50/100 by 11/37/2/10 samples, with raw peaks
1.0558/1.1291/1.0089/1.0783. The pipeline clamped precisely those samples.
The 2e-5 control at step 50 also exceeded unit amplitude for 6 samples.
Therefore the clipped peaks are real VAE output events, not FLAC artifacts,
but they are sparse and not exclusive to the high-LR arm. The optimized
SongEval mean fell at step 100, so these observations do not establish that
SongEval rewards clipping.

Two initial launches OOMed when another legitimate 5090 experiment grew its
GPU allocation; neither produced a public post-update-3 evaluation. The
run resumed from its saved step-3 adapter and AdamW state after that
other process exited. It reached checkpoint 25, temporarily released the GPU,
then resumed with the same AdamW state and evaluated checkpoints 50 and 100. Lower
held-out scores and sparse full-scale peaks are not, by themselves, evidence
of reward hacking.
