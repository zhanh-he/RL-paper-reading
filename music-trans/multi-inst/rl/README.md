# Multi-instrument post-training

Use [DPO](dpo/README.md) for same-input preference pairs and [GRPO](grpo/README.md) for grouped candidate rollouts. Both require a frozen instrument-aware test set and baseline checkpoint. Add one directory per real run under the method, following the [experiment record](../../../notes/EXPERIMENT_RECORD.md). A future optimizer gets a sibling directory when implemented.
