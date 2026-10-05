# YuE2 demo result recovery (2026-10-05)

The [public replay](https://zhanh-he.github.io/RL-paper-reading/demos/#lyrics)
now binds five completed YuE2 arms to their archived receipts and 0/1/5/25/50/100
held-out audio for three fixed prompts (seeds 5101-5103). New audio and images
are in `platform/site/demos/audio/` and `visuals/`; the build checks the
receipt's audio SHA-256 whenever available. SongEval and MuseCritic use
different scoring models, so their values must not be compared numerically.

| Arm | Reward at 0 | Reward at 5 | Reward at 25 | Reward at 50 | Reward at 100 | Offline KL at 100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SongEval, 2e-5 | 3.8403 | 3.6160 | 3.5629 | 3.5704 | 3.6063 | 0.000791 |
| SongEval, 1e-4 | 3.8403 | 3.4285 | 3.6296 | 3.7639 | 3.4296 | 0.000874 |
| SongEval, 1e-3 | 3.8403 | 3.6504 | 3.7188 | 3.6722 | 3.0909 | 0.497705 |
| SongEval, 1e-2 | 3.8403 | 4.0533 | 1.5803 | 1.5034 | 1.7783 | 21.022778 |
| MuseCritic PCM24, 2e-5 | 2.9677 | 2.6417 | 3.0177 | 3.1146 | 2.2443 | 0.000795 |

The table reports three-prompt held-out reward means, not training reward or
human preference. The common SongEval step-1 mean is 3.6030; the same audio
scores 2.5385 under canonical-FLAC MuseCritic. The 1e-2 SongEval arm's
short-lived step-5 rise was followed by reward collapse and large offline
policy drift. The 1e-3 arm also drifted by step 100 and fell below its
step-1 reward. MuseCritic briefly exceeded its own step-0 reward at steps
25 and 50, then fell below it at step 100. The low-LR SongEval arms show no
durable held-out reward increase. These observations do not establish why
the low-LR runs stagnated or whether any clip sounds better.

The left chart uses five-step windows of **training rollout** five-dimension
mean reward, while the right uses **offline** mean token
`D_KL(step-1 reference || checkpoint)` on three fixed, reference-generated
semantic trajectories. It does not plot online GRPO KL, which was neither
constrained nor logged. The shared step-1 LoRA is the KL zero, not the
unadapted step-0 model. Source audits:
[SongEval KL](../2026-09-30-yue2-kl-audit/README.md),
[MuseCritic PCM24 KL](../2026-10-01-yue2-musecritic-pcm24/README.md),
[high-LR stress](../2026-09-30-yue2-high-lr/README.md).

Training used eight original hand-written prompts, not SongEval,
WildSongBench, or CMI-RewardBench training examples. Every held-out clip
hit the 600 semantic-token cap, roughly 24 seconds. The scorer models and
audio format sensitivity, small sample count, no explicit KL or duration
constraint, and lack of blind human ratings all limit interpretation.
