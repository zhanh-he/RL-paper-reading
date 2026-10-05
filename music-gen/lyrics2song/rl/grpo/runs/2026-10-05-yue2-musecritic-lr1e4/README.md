# YuE2 MuseCritic PCM24-FLAC GRPO, learning rate 1e-4

This arm completed 100 optimizer steps on lab5090. It used the shared step-1
YuE2 LoRA, eight original training prompts, three disjoint fixed held-out
prompts (seeds 5101-5103), two samples per group, and a 600 semantic-token
cap. `experiment.json` records the exact prompt text and settings.

| Step | Three-song held-out MuseCritic mean |
| ---: | ---: |
| 0 | 2.967708 |
| 1 | 2.538542 |
| 5 | 2.413542 |
| 25 | 2.848438 |
| 50 | 2.659375 |
| 100 | 2.732292 |

`steps.jsonl` contains 99 ordered updates (steps 2-100). The six held-out
receipts contain 18 scored PCM24-FLAC clips; each published audio file was
checked against its receipt SHA-256 and the scorer's raw five-dimension result.
Step 0 and step 1 reuse the same frozen audio and canonical-format scores as
the [2e-5 arm](../2026-10-01-yue2-musecritic-pcm24/README.md).

The step-100 mean is 0.193750 above step 1 but 0.235416 below the base-model
step 0. This small, fixed held-out set does not establish perceptual quality
or generalization. The offline fixed-reference KL probe is not yet available
for this arm; the public demo labels it as pending rather than drawing a curve.
All held-out generations reached the semantic-token cap, so duration and
truncation remain limitations. There is no explicit training KL, lyric-match,
loudness, or clipping constraint.
