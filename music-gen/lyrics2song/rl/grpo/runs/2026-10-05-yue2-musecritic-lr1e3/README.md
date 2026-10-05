# YuE2 MuseCritic PCM24-FLAC GRPO, learning rate 1e-3

Partial snapshot for the public demo on 2026-10-05. The lab5090 run was
still training when these receipts were copied. `experiment.json` records
the same eight original training prompts and three disjoint held-out prompts
as the other MuseCritic learning-rate arms.

| Step | Three-song held-out MuseCritic mean | Status |
| ---: | ---: | --- |
| 0 | 2.967708 | Shared frozen base |
| 1 | 2.538542 | Shared source LoRA |
| 5 | 2.448437 | Scored PCM24-FLAC replay archived |
| 25, 50, 100 | - | Training or evaluation not recovered |

The nine scored FLACs represented by these receipts are bound to the audio
in the public demo by SHA-256. The 0/1 clips are the same shared origin as
the 2e-5 arm. No complete training trace, final checkpoint, or offline KL is
claimed here. Missing values are reserved as explicit placeholders. In
particular, the step-5 difference is not a learning-curve conclusion.
