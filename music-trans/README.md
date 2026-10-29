# Music transcription

[AMT literature catalog](https://zhanh-he.github.io/RL-paper-reading/amt/) · [AMT live experiments](https://zhanh-he.github.io/RL-paper-reading/amt/demos/) · [AMT update requests](../upd_request/amt/) · [AMT source table](../platform/data/amt/literature.csv)

The [AMT experiment page](https://zhanh-he.github.io/RL-paper-reading/amt/demos/) has its own experiment manifest, training traces, paired replay, checkpoint metrics and Dataset Protocol / Models / Rewards sections. Its next experiment is undecided; the manifest is intentionally empty. [Publishing instructions](../platform/docs/AMT_EXPERIMENTS.md) explain how to attach actual run receipts later. Existing Choral and MuScriptor replay stays available in the historical demo.

Transcription experiments turn audio into symbolic notes. [Multi-instrument](multi-inst/README.md) measures instrument-aware note detection; [choral singing](choral-singing/README.md) measures SATB voice assignment and note boundaries. Both need source-disjoint train/dev/test splits and note-level independent evaluation. They share a task family, not a reward formula: an instrument label is not automatically an S/A/T/B voice label.

Each task has its own model adapters, rewards and DPO/GRPO run records. Published MuScriptor results are background evidence, not a local result in this repository.
