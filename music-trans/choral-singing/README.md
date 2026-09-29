# Choral singing transcription

**Contract:** a choral recording in, four named S/A/T/B note tracks out. The primary endpoint is voice-aware note F1 at a declared onset tolerance, with frame/offset scores, voice confusion and weak-part performance alongside it. Instrument-labelled output does not become SATB output without a validated mapping.

- [Models](models/README.md) contain public-safe provenance and adapters only.
- [Rewards](rewards/README.md) distinguish actual voice coverage from the mistaken goal of continuous four-part activity.
- [RL](rl/README.md) keeps DPO/GRPO run records separate.

The most useful first test is a small, source-disjoint set with aligned voice MIDI, not a large pool of weak labels. No local post-training result is published here yet.
