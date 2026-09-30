# SATB transcription rewards

The implemented [SATB reward](satb_reward.py) scores four labeled MIDI tracks
against reference notes. The complete-note endpoint matches pitch, SATB voice,
onset within 50 ms and offset within `max(50 ms, 20% of reference duration)`.
Matching is one-to-one, so duplicate predictions cannot each collect recall.
Frame F1 uses a 10 ms grid. [Metric tests](test_note_metrics.py) cover boundary,
track-assignment and duplicate-note behavior. The per-voice onset and
onset+offset F1 on all nine public checkpoint MIDI outputs were also
[cross-checked against `mir_eval` 0.8.2](validate_mir_eval.py) with the same
50 ms tolerances.

| Component | Weight | Why it is present | Single-reward trap |
| --- | ---: | --- | --- |
| SATB onset F1, macro | 0.25 | Pitch, onset and voice assignment | Every note can be too short |
| SATB onset+offset F1, macro | 0.25 | Complete notes | Sparse easy notes can dominate |
| SATB frame F1, macro | 0.15 | Sustained pitch occupancy | Long notes can be fragmented |
| Active-voice F1, 0.5 s windows | 0.10 | Voice presence/coverage | Extra octaves inside a present voice go unseen |
| Minimum per-voice complete-note F1 | 0.10 | Prevent sacrificing a quiet part | Can overfit a weak track's few notes |
| SATB complete-note precision | 0.10 | Penalize added notes | Deleting Bass retains precision |
| `1 - fragmentation rate` | 0.05 | Preserve sustained notes | Silence scores 1 |

The combined score is a weighted sum, not a proof against reward hacking. The
[constructed audit](audits/2026-09-30-satb-reward-counterexamples.json) changes
one original 32-note SATB reference song at a time: all-short notes, octave
overfill, sustained-note fragmentation, deleted Bass, swapped S/A and silence.
It tests each component's blind spot and whether the combination lowers that
counterexample's score. These are **not** outputs of a GRPO policy. They are
rendered with fixed ACE Studio singers on the [demo page](../../../platform/site/demos/).

The separate [event-head GRPO experiment](../rl/grpo/runs/2026-09-30-event-head/README.md)
actually trains single-reward and combined policies on the same split. A
single public-synthetic replay is an additional listening check, not a held-out
mean. Do not claim that combining rewards guarantees better generalization;
compare equal budgets, complete-note F1 and per-part failures before making
that statement.

An audio-to-MIDI consistency reward is a future auxiliary experiment. It was
not implemented or used in the receipts here; a renderer can introduce its own
timbre shortcuts. The four-part condition must be conditional on the reference
being active, not a blanket reward for filling all four tracks through rests.
