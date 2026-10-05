# YuE2 MuseCritic GRPO with canonical PCM24 FLAC reward

This is a separate correction of the earlier [float-WAV-scored run](../2026-09-30-yue2-musecritic/README.md).
The official MuseCritic model scored **the exact PCM24 FLAC bytes retained
for replay**, not a temporary float WAV. The eight original hand-written
training prompts, three original held-out prompts, fixed seeds, shared
one-step YuE2 LoRA, 2e-5 learning rate, group size two, and 600 semantic-token
cap were unchanged. See `experiment.json` for every prompt and setting.

| Optimizer step | Canonical FLAC MuseCritic mean | Old float-WAV mean |
| ---: | ---: | ---: |
| 0 (base model) | 2.9677 | 2.7542 |
| 1 (shared source LoRA) | 2.5385 | 2.8396 |
| 5 | 2.6417 | 2.8948 |
| 25 | 3.0177 | 2.3510 |
| 50 | 3.1146 | 2.8229 |
| 100 | 2.2443 | 2.5583 |

Each value averages **three** fixed-seed held-out clips, not a population
estimate. The corrected run passed a two-step smoke test and completed step
100 with adapter and optimizer checkpoints. All six receipts (18 songs) were
checked against the saved FLAC SHA-256 and the scorer's raw five-dimension
`result.json`; every binding matched. Step-0 and step-1 FLACs are the same
source audio as the old run, and their canonical scores match the independent
[cross-reward audit](../2026-09-30-yue2-cross-reward/README.md) exactly.

On this tiny held-out set, step 25 and 50 were 0.4792 and 0.5760 above the
source LoRA; step 100 was 0.2943 below it. Two of the three songs improved at
step 50 and one fell slightly. The four roughly 25-update training windows
had reward means 2.6486, 2.6917, 2.7044, and 2.6826, without convincing
monotonic improvement. This is a **transient checkpoint-level observation**,
not evidence of sustained post-training benefit or audible quality gains.
The old and corrected arms optimize different audio representations after
step 1, so their later checkpoint scores should not be read as paired
measurements of the same waveform.

All held-out clips hit the 600-token semantic cap (about 24 seconds).
Near-full-scale sample fractions remained tiny, even where a peak touched
1.0. There is no evidence here of persistent clipping-based reward hacking.
The scorer's severe [WAV/FLAC sensitivity](../2026-10-01-yue2-musecritic-format-audit/README.md),
small prompt set, group-two rank-only advantages, absent reference KL, and
short output cap remain important limitations. Blind listening and more
held-out samples are needed before any music-quality claim.

An additional 2026-10-05 fixed-reference audit is in `kl.json`. Its
same-weight adapter identity check returned `[0, 0, 0]`. Conditional KL
against the shared step-1 LoRA was 0.000792, 0.000802, 0.000797, and
0.000795 at steps 5/25/50/100. It uses the same three reference-generated
semantic trajectories and codec-constrained definition as the SongEval
[KL audit](../2026-09-30-yue2-kl-audit/README.md). These are **offline**
measurements, not an optimizer KL penalty or logged on-policy training KL.
