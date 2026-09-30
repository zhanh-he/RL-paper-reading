# Cross-scoring SongEval stress-test audio with MuseCritic

The official MuseCritic checkpoint scored the exact **PCM24 FLAC files**
archived by both SongEval high-LR runs, with a single model load and the
same official critique prompt for all 36 clips. Each value is a three-song
mean. It is on MuseCritic's scale, not SongEval's.

| Step | 1e-3 MuseCritic | 1e-2 MuseCritic |
| ---: | ---: | ---: |
| 0 | 2.9677 | 2.9677 |
| 1 | 2.5385 | 2.5385 |
| 5 | 2.7479 | 2.3354 |
| 25 | 2.6438 | 1.7807 |
| 50 | 2.6615 | 1.4505 |
| 100 | 2.0104 | 1.8786 |

The shared baseline FLACs are byte-identical across arms and have identical
MuseCritic scores in this batch. The 1e-2 arm's step-5 SongEval mean rose
from 3.6030 to 4.0533, but its MuseCritic-on-FLAC mean fell from 2.5385 to
2.3354. This is disagreement between evaluators, **not** proof of audible
reward hacking. MuseCritic itself is highly sensitive to nearly inaudible
WAV-to-FLAC quantization in a separate
[format audit](../2026-10-01-yue2-musecritic-format-audit/README.md), so
these scores should not substitute for blind listening.

Each `result.json` contains the five raw dimensions for one fixed clip.
The scoring script is `../../run_yue2_cross_musecritic.sh`.
