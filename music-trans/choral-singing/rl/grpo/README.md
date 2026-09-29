# Choral singing GRPO

The [frame-head experiment](runs/2026-09-29-frame-head/README.md) trained
ChoralStream for 1,152 optimizer updates using 96 original training songs and
evaluated on 30 held-out songs. Its frame-head F1 rose from 0.1378 to 0.3033;
a supervised BCE control reached 0.4541. Three GRPO seeds and the held-out
protocol are recorded with that run.

This is **not an end-to-end transcription gain**. The event decoder does not
consume the trained frame head, so baseline and GRPO decode to identical MIDI.
The [50 ms note receipt](runs/2026-09-29-frame-head/note_metrics_50ms.json)
reports unchanged pitch-only onset F1 0.1063 and onset+offset F1 0.0234;
SATB-track-aware F1 is 0.0661 and 0.0145. The [original synthetic SATB
demo](runs/2026-09-30-public-synthetic/receipt.json) publishes input,
reference, baseline and post-training audio/MIDI for that exact comparison.

The next GRPO policy must control event decoding itself. Use reference-backed
50 ms onset and onset+offset F1 as primary rewards; inspect voice-aware scores,
continuity and occupancy as guardrails. Four-track occupancy alone can reward
spurious notes. Record grouped candidate rewards, variance, KL and weak-voice
F1 before claiming a new experiment.
