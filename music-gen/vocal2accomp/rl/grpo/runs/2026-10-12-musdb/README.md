# MUSDB18-HQ dispatch receipt

Verified 2026-10-12 03:29 UTC. This is a scheduler receipt, not a result table.
See [dispatch.json](dispatch.json) and the [protocol](../../DATA-PROTOCOL.md).

The full corpus download is running. Preparation, a two-update real GPU
preflight, and two main training seeds have been submitted in dependency order.
Six reward ablations follow the basic seed13 run. The RTX5090 is excluded.
Dependency-held GPU jobs are not yet executing optimizer updates.

The independent reward environment passed an original Beat-v5/Coverage scoring
smoke on Gadi. Kaya completed 16 tests and the accompanying
[constructed function audit](constructed-reward-audit.json). That audit uses
synthetic signals, not LaDA generations, and exposes a remaining harmonic-fit
weakness of the combined reward.

A separate [persistent-worker smoke](reward-worker-smoke.json) passed two
requests with the actual scoring formats: PCM24 44.1 kHz input and FLOAT stereo
48 kHz candidate (Gadi job181038322, exit0).

Actual eligible songs, phrases and minutes will be produced by
`prepared/protocol.json` on Gadi. Actual update latency, adapter changes and
validation replays will be recorded by the trainer. Neither counts nor quality
improvements are inferred from a queued job.
