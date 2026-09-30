# Fixed vocal to accompaniment

**Contract:** the provided singing voice is immutable; the model supplies accompaniment aligned to that voice. Some models output accompaniment only, others complete a full mix. The adapter must expose the generated accompaniment or a documented separation step before comparing systems. Never treat a changed generated vocal as an improvement to the accompaniment.

- [Models](models/README.md) track LaDA-Band as the preferred target and public ACE-Step and AnyAccomp baselines without copying weights.
- [Rewards](rewards/README.md) are beat, coverage and richness as independent arms, followed by a guarded combination.
- [RL](rl/README.md) separates DPO pair learning from GRPO online sampling.

Every run must keep input vocals fixed across baseline and post-training outputs. Source/song-level splits, redistribution rights and blind rhythmic-fit listening matter as much as proxy reward scores. The [ACE-Step inference demo](models/ace-step-1.5/README.md) verifies local generation only; no local vocal2accomp post-training result is published here yet.
