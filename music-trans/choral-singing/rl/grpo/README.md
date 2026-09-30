# Choral singing GRPO

**Current event-level run:** [16-song/8-song GRPO pilot with 0/100/300/1000
checkpoints](runs/2026-09-30-event-head/README.md). This trains ChoralStream's
MIDI event heads and changes decoded notes. The [SATB reward design and
counterexamples](../../rewards/README.md) separate actual policy ablations
from constructed failure cases. A fixed-singer ACE Studio listening replay is
on the [demo page](../../../../platform/site/demos/).

## Earlier diagnostic run

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

The event-level pilot above addresses the decoder boundary. The frame-head
experiment remains useful as a negative control: optimizing a metric in a head
that the MIDI decoder does not use cannot change the final transcription.
