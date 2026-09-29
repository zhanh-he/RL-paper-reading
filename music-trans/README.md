# Music transcription

Transcription experiments turn audio into symbolic notes. [Multi-instrument](multi-inst/README.md) measures instrument-aware note detection; [choral singing](choral-singing/README.md) measures SATB voice assignment and note boundaries. Both need source-disjoint train/dev/test splits and note-level independent evaluation. They share a task family, not a reward formula: an instrument label is not automatically an S/A/T/B voice label.

Each task has its own model adapters, rewards and DPO/GRPO run records. Published MuScriptor results are background evidence, not a local result in this repository.
