# MUSDB18-HQ formal Vocal2accomp protocol

Status: implementation and cluster staging on 2026-10-12. The complete corpus is
being downloaded. No new training result is claimed before a successful GPU
preflight and recorded optimizer updates.

## Data and scope

- Source: <https://zenodo.org/records/3338373>, original uncompressed stereo
  44.1 kHz WAV archive, 22,656,664,047 bytes. Official MD5:
  `12d4f2ecd55245a4688754dd76363103`.
- Academic research use; original stems and checkpoints stay outside Git and
  public Pages. Any public playback needs a separate per-track release check.
- Official MUSDB tool validation song list: 86 train / 14 validation; official
  50 test songs sealed. The configuration and manifests are SHA-256 recorded.
- Test preparation reads file headers and inventories names, never test PCM.
  No test excerpts are selected or scored during training/checkpoint selection.
- Native input = original vocals. Reference backing = drums + bass + other.
  The reference is one valid arrangement, not the unique correct waveform.
- Every non-overlapping 12-second clip passing quality/activity criteria is
  retained; no 20-song pilot subset or hand-picked musical success subset.
- Vocal activity >=35%, accompaniment activity >=50%, original vocal RMS
  >=0.003, finite/aligned stereo sources, vocal clipped fraction <=0.1%.
  Activity uses 40 ms stereo-energy frames and an RMS threshold of 0.003.
- Known official source-erratum recordings are excluded from the clean pool,
  with every excluded clip/reason recorded. Original recordings remain intact
  for a future challenge evaluation. This is not a guarantee of zero bleed.
- A common gain is applied to vocal and backing: min(vocal RMS target 0.04,
  maximum vocal/backing/reconstructed-mixture peak target 0.90, maximum gain 4).
  There is no independent per-stem peak normalization. Prepared vocals use
  PCM24 for Madmom compatibility; backing and generated raw outputs use FLOAT.
- `protocol.json` records actual eligible song/clip/minute counts, stem activity
  counts, artist overlaps, known errata and exclusions. Counts are not invented
  while the download is incomplete.
- Drums/bass/other activity is a coarse arrangement census. `other` is not an
  instrument annotation; neither spectral bands nor this census establish
  guitar/piano/strings/brass completeness. Keep neutral arrangement prompts,
  retain all qualifying source genres, and use fine-stem MoisesDB as a later
  external expansion after duplicate and access audits.
- These are studio vocal stems, not guaranteed dry solo home recordings.
  Artist-disjoint and pretraining-unseen claims are not justified by song splits.

## Training

`train_lada_formal.py` reuses the previously exercised LaDA diffusion trajectories,
action log probabilities and output-head LoRA update. It is single-pass
group-relative policy-gradient training with sampled KL, not a validated
multi-epoch PPO implementation. Ratio clipping is effectively inactive in the
single optimizer pass. Frozen backbone/codec/CLaMP3; rank-32 output-head LoRA.

Main arm: BCR-v1, two seeds (13, 29), group size 4, 16 denoising iterations,
AdamW learning rate 1e-4, reference KL coefficient 0.02, two complete data epochs.
Every clip is visited once per epoch with shuffled song round-robin ordering.
Long songs can have more total clips, but cannot monopolize early training.
All conditions use the same neutral instrumental prompt; no artist/title prompt.

Save at 0/5/50/100/300/1000 updates, each 300 updates, every epoch boundary and
completion. Save on the 22-hour runtime budget too; later scheduler blocks resume
the full adapter/optimizer state. A 100/300-step checkpoint is not corpus coverage.
Record each condition, elapsed time, group component scores, reward std/ties,
adapter norm, sampled KL, VRAM and rollout count. The first GPU preflight measures
real update latency; no timing estimate from a different GPU is a promised ETA.

This run also changes the mock's rank, group size, segment duration, reward and
learning rate. A better outcome would not isolate a causal dataset effect.
A later small-subset/full-pool ablation must hold those settings fixed.

## Rewards

All three components are project-developed, with external Madmom as the Beat-v5
detector backend. No component is an off-the-shelf validated perceptual judge.

- B: original project Beat-v5, Madmom backend, sigma=55 ms, subdivision=2,
  12-second windows. Keep confidence, abstention and components; abstaining B
  contributes zero with its raw score still logged.
- C: original project STFT activity Coverage, -40 dB threshold, n_fft=2048,
  hop=512. It is not the previous trainer's 40 ms RMS proxy. Training term is
  `Ccap = min(1, C/0.7)`; raw STFT coverage and old RMS diagnostic both logged.
- R: existing Richness-v0 spectral-layer activity/movement/tonality proxy,
  before its old vocal-loudness penalty. It is explicitly not instrument count
  or a validated human judgment. Detuned/unrelated layers remain a known risk.
- Q: stereo peak above 0.9, clipped samples >=0.999 and spectral-flatness above
  0.35 penalties. No normalization before this safety check.
- L: 0.025 times the absolute RMS dB distance beyond a +/-6 dB band around the
  original backing. This is reference-informed mix calibration, not exact
  arrangement reconstruction or LUFS normalization.

`reward = 0.50 B + 0.25 Ccap + 0.25 R - Q - L`.

Weights/caps/guardrails are fixed before main training. Do not claim that the
combination prevents reward hacking: constructed attacks and actual model
rollouts must be reported separately, and blind listening remains necessary.

Kaya job134085 completed a constructed function audit and 16 unit tests. White
noise scored -0.016 and clipped chord -1.583, while a drone still scored 0.425.
Detuned layers scored 0.503 versus 0.498 for the envelope-following chord:
BCR-v1 has no validated harmonic-fit constraint. These are synthetic signals,
not LaDA outputs or evidence that training learned these attacks. The audit
uses Kaya FFmpeg8; matched model runs use Gadi FFmpeg4.3.1. Preserve environment
versions and do not treat cross-platform values as byte-identical.

## Ablations after the basic run

Use the same frozen initialization, seed13, conditions/order, sampler, group4,
LoRA scope, optimizer and Q/L guardrails. One complete data epoch per ablation;
compare against the main arm's first-epoch checkpoint, not its second epoch.
Each ablation has two 24-hour budget blocks with checkpoint/resume. An incomplete
epoch must be labeled partial and is not an equal-data comparison; the scheduler
budget does not guarantee completion.

| Arm | B | Ccap | R | Question |
| --- | ---: | ---: | ---: | --- |
| bcr | .50 | .25 | .25 | Basic combined objective |
| no_beat | 0 | .50 | .50 | Does removing rhythmic fit hurt alignment? |
| no_coverage | .67 | 0 | .33 | Does removing activity allow sparse outputs? |
| no_richness | .67 | .33 | 0 | Does removing layer proxy simplify arrangements? |
| beat | 1 | 0 | 0 | Beat-only failure cases under shared safety checks |
| coverage | 0 | 1 | 0 | Sustained activity / drones / saturation |
| richness | 0 | 0 | 1 | Excess/unrelated layers despite proxy gain |

Guardrails are shared, so these are task-component ablations, not reproductions
of the old unguarded mock arms. A separate later guardrail-on/off study can test
the safety mechanism without confounding this matrix.

## Evaluation and replay

- Fixed, earliest qualifying validation phrase per eligible validation song.
  Fast trajectories: first three such songs, seeds777/778; epoch/final checkpoints:
  all eligible validation songs with both seeds. Counts/IDs come from manifests.
- This is a validation-probe protocol, not exhaustive validation or official
  test evaluation. Full validation and sealed-test evaluation are separate runs.
- Store input vocal, reference backing, frozen baseline and adapted backing
  IDs/checksums. Mix each generated backing with the unchanged input vocal.
- Report raw B/C/R and Q/L separately, abstention rate, audio-level clipping,
  relative RMS and sampled training KL. These are diagnostic scores, not
  independent evidence of improved music when the same terms are optimized.
- Primary listening questions: rhythmic/harmonic fit, musicality, useful
  arrangement diversity, audio artifacts and vocal masking. Randomized blind
  A/B with identical vocal and listening normalization; retain raw audio too.
- Aggregate by song, with song-level uncertainty. More clips/seeds from the
  same song are not more independent songs. Select checkpoints using validation
  and listening, never test scores or best-looking presentation examples.

## Run locations

Gadi: `/scratch/wa66/hm6416/vocal-musdb-20261012`.
Kaya reward audit: `/group/ems011/zhe/vocal-musdb-20261012`.
RTX5090 is excluded from this run. `runs/2026-10-12-musdb/dispatch.json` stores
the actual scheduler submissions and their latest verified state.
