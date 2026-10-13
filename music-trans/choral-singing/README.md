# Choral singing transcription

**Contract:** a choral recording in, four named S/A/T/B note tracks out. The primary endpoint is voice-aware note F1 at a declared onset tolerance, with frame/offset scores, voice confusion and weak-part performance alongside it. Instrument-labelled output does not become SATB output without a validated mapping.

- [Models](models/README.md) contain public-safe provenance and adapters only.
- [Rewards](rewards/README.md) distinguish actual voice coverage from the mistaken goal of continuous four-part activity.
- [RL](rl/README.md) keeps DPO/GRPO run records separate.
- [Research map](research/README.md) connects SATB datasets, direct and adjacent models, and reward/evaluation choices.

The [event-head GRPO run](rl/grpo/runs/2026-09-30-event-head/README.md) has 16 training and 8 unseen songs with fixed 5.12 s clips and same-origin reward ablations. It is not a full-song or cross-dataset result; keep the earlier frame-head pilot and the public synthetic listening replay separate.
