# SATB rewards

| Proposed file | Signal | Failure to test |
| --- | --- | --- |
| `voice_note_f1.py` | Pitch/onset/offset matched within the correct S/A/T/B voice | Strong soprano masking weak alto/tenor |
| `four_part_coverage.py` | Presence/recall of voices **when the GT says they sing** | Rewarding four tracks through rests and solos |
| `continuity.py` | Unwarranted note fragmentation and chunk discontinuity | Merging real breaths or repeated notes |
| `audio_consistency.py` | MIDI-render versus input alignment, auxiliary only | Timbre and renderer shortcuts |

Coverage must be paired with false-note/false-voice precision. Keep note F1 and voice assignment as independent endpoints even if a coverage reward is trained. One module per simple reward is sufficient; a future `combine.py` belongs beside them after single-reward ablations.
