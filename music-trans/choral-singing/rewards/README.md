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

## What each single-reward arm tests

Each arm trains from the same frozen event-head checkpoint and uses only the
named component. The 300-step public replay shows its decoded MIDI on the same
song as the baseline and combined arm; the held-out table reports both 300 and
1,000 steps. These are distinct observations: one listening example is not the
mean over the eight unseen songs.

- **Onset** asks whether the right pitch begins in the right SATB track within
  50 ms. It does not measure how long that pitch lasts, so clipped notes can
  still be rewarded.
- **Onset + offset** additionally requires the note ending within the larger
  of 50 ms or 20% of its reference duration. It directly rewards complete
  notes, but a model can favor a small set of easy notes while missing a quiet
  part; inspect per-voice recall alongside the macro score.
- **Frame** scores pitch occupancy on a 10 ms grid. It can improve sustained
  activity while leaving onset timing and note segmentation poor; splitting
  one long note into many fragments is a concrete failure mode.
- **Coverage** compares which of the four voices are active in each 0.5 s
  window. It detects a missing part but ignores wrong pitches and extra
  octaves within an already-active part. It does not require all four voices
  during rests.
- **Continuity** rewards one minus the sustained-note fragmentation rate. It
  is deliberately weak evidence by itself: silence contains no fragmented
  sustained note and can score well. It needs an activity or note-match term.
- **Weak voice** uses the lowest complete-note F1 among S, A, T, B, to guard
  against sacrificing the hardest voice. The recorded 300-step arm had zero
  effective policy updates, so its public A/B replay is unchanged; this is an
  observed optimization failure, not evidence that the component is useless.
- **Precision** measures the share of predicted SATB complete notes that
  match reference notes. Deleting hard Bass notes can preserve precision while
  destroying recall, so precision must be paired with coverage or F1.

The combined objective weights onset and complete-note F1 most heavily,
while frame, active-voice coverage, minimum-voice F1, precision and continuity
constrain their individual shortcuts. Whether those constraints help is an
empirical question: compare the equal-budget held-out results and inspect the
decoded MIDI, rather than assuming the weighted sum prevents hacking.

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
