# Multi-instrument transcription

**Contract:** mixed audio in, instrument-labelled MIDI events out. The primary endpoint is instrument-aware note F1; report onset, offset, frame and per-instrument precision/recall separately. A pitch-correct note assigned to the wrong instrument is not a success.

- [Models](models/README.md): one adapter/config folder per real model, starting with MuScriptor.
- [Rewards](rewards/README.md): GT-backed note/boundary rewards first; audio consistency is an auxiliary check.
- [RL](rl/README.md): DPO and GRPO runs kept separate with matched base checkpoints and data splits.

Do not call a model's published RL improvement a result from this repo. Public
synthetic paired audio/MIDI can validate the pipeline. A local
[MuScriptor-medium GRPO smoke](rl/grpo/runs/2026-09-30-muscriptor-medium-smoke/README.md)
and anonymous 8-song short-excerpt audit now exist, but neither measures
instrument-aware full-song generalization. That needs a disjoint train/test
study with usable rights and track-labelled ground truth.
