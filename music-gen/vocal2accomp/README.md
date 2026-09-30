# Fixed vocal to accompaniment

**Contract:** the provided singing voice is immutable; the model supplies accompaniment aligned to that voice. Some models output accompaniment only, others complete a full mix. The adapter must expose the generated accompaniment or a documented separation step before comparing systems. Never treat a changed generated vocal as an improvement to the accompaniment.

- [Models](models/README.md) track LaDA-Band as the preferred target and public ACE-Step and AnyAccomp baselines without copying weights.
- [Rewards](rewards/README.md) document the measured fast combination, coverage-only and original Beat-v2 arms, plus the still-unvalidated richness target. Do not rename the onset/band proxies to Beat-v2/richness.
- [RL](rl/README.md) separates DPO pair learning from GRPO online sampling.

Every run must keep input vocals fixed across baseline and post-training outputs. Source/song-level splits, redistribution rights and blind rhythmic-fit listening matter as much as proxy reward scores. The [ACE-Step inference demo](models/ace-step-1.5/README.md) verifies local generation only. LaDA-Band now has a measured one-singer online GRPO-style run through step 300, with [paired audio, spectrograms and independent metrics](../../platform/site/demos/vocal-lada.html), plus a replay of an untrained phrase from the same recording. A separate coverage-only reward ablation is running. The [GRPO note](rl/grpo/README.md) distinguishes actual reward optimization from offline diagnostics and its single-pass objective from full multi-epoch GRPO.
