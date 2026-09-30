# YuE2 MuseCritic GRPO, 100-step pilot

This 5090 run started from the same one-step YuE2 LoRA used by the SongEval
arms. The eight training and three held-out style/lyric prompts are original,
hand-written, and identical across reward arms; none came from SongEval,
WildSongBench, or CMI-RewardBench. The exact requests and settings are in
`experiment.json`. The official MuseCritic checkpoint and inference code
produced a critique plus five reward scores for each generated clip; their
mean was the sole training reward. These scores are **not** on SongEval's
scale, even though both outputs use five similarly named keys.

| Optimizer step | Held-out MuseCritic mean |
| ---: | ---: |
| 0 (base model) | 2.7542 |
| 1 (shared source LoRA) | 2.8396 |
| 5 | 2.8948 |
| 25 | 2.3510 |
| 50 | 2.8229 |
| 100 | 2.5583 |

Each cell averages three fixed-seed held-out songs. The two-step smoke test
passed and the run completed with step-100 adapter and optimizer checkpoints.
The step-100 held-out score is 0.2813 below the source LoRA, despite the mean
training reward rising from about 2.61 in updates 2-26 to 2.79 in updates
77-100. Training windows use changing prompts and fresh samples, so that
rise is not a matched-pair estimate of generalization. All held-out clips
hit the 600-token cap, and none had near-full-scale samples. The result does
not demonstrate successful MuseCritic post-training or reward hacking.

**Reward-format caveat:** this pilot scored the temporary float WAV but kept
PCM24 FLAC for replay. The [format audit](../2026-10-01-yue2-musecritic-format-audit/README.md)
found that essentially identical WAV/FLAC signal values can receive sharply
different MuseCritic scores. The numbers above describe the temporary WAVs,
not a robust property of the replay FLACs. A separate canonical-FLAC rerun
was started; do not present this pilot as a validated quality improvement.

This run used the same group-of-two, single-update custom PyTorch objective
as the SongEval arms. It had no reference KL, explicit clipping/duration
guardrail, or lyric-fidelity reward. The [KL audit](../2026-09-30-yue2-kl-audit/README.md)
covers the SongEval arms, not these MuseCritic checkpoints; it is not this
run's training-time KL.
