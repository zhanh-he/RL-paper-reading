# Vocal-to-accompaniment post-training

Run beat, coverage and richness **one at a time** from the same starting policy and fixed-vocal inputs before trying `combine.py`. [DPO](dpo/README.md) needs same-vocal preference pairs; [GRPO](grpo/README.md) needs same-vocal grouped rollouts. Keep each run's reward version, duration, seeds, compute and independent listening record using the [experiment template](../../../notes/EXPERIMENT_RECORD.md).
