# Multi-instrument rewards

Prefer small, separately testable modules when the definitions are validated:

| Proposed file | Signal | Required guardrail |
| --- | --- | --- |
| `onset_frame_offset.py` | GT-aligned note start, sustained frames and end | Do not hide instrument swaps |
| `instrument_f1.py` | Instrument-aware one-to-one note matching | Inspect per-instrument precision and rare classes |
| `audio_consistency.py` | Rendered MIDI versus source audio | Renderer/timbre bias; never replace GT F1 |
| `continuity.py` | Unsupported fragmentation and chunk breaks | Preserve real rests and repeated-note attacks |

`humanization.py` is only appropriate for a separately defined expressive-timing/dynamics target with labels or a qualified human comparison. Exact transcription should not reward arbitrary timing deviations from ground truth. Keep each simple reward in one Python file; add `combine.py` only after single-reward behavior and scale are audited. No implementation is claimed by this README.
