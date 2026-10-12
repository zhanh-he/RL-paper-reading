# Formal Recovery: 2026-10-12

These results use the pinned CMI-Pref text+lyrics projection: 270 training
conditions, 30 fixed validation conditions and 121 sealed official-test
conditions. They are not the earlier eight-prompt mock experiments. Reference
audio is not supplied to either model. The three replay conditions are fixed
validation indexes 0, 10 and 20, seeds 5101, 5111 and 5121.

## Verified Training

- YuE2-SongEval: lab5090 BF16, learning rates 2e-5, 1e-4 and 3e-4;
  all 270 optimizer steps and all fixed 30-condition validation checkpoints
  were independently audited. Gadi BF16 repeats are reported separately.
- YuE2-MuseCritic: Gadi BF16, the same three learning rates; all 270 steps
  and fixed validation checkpoints were audited. MuseCritic scores the exact
  archived PCM24-FLAC, correcting the earlier WAV-versus-FLAC mismatch.
- Muse-MuseCritic: Gadi v2, 1e-6 and 3e-6 completed 270 steps. The 1e-5 arm
  stopped after 140 successful steps when a rollout contained no AUDIO tokens;
  its last saved checkpoint is step 120.
- Muse-SongEval: Gadi v2 stopped after 76 / 142 / 32 successful steps at
  1e-6 / 3e-6 / 1e-5 respectively. Very short generated audio caused a MuQ
  input-padding failure in the first two arms; the third produced no AUDIO
  tokens. Their last saved checkpoints are 60 / 120 / 30.
- The six old Muse arms are superseded and retained only for audit. All eight
  pending validation jobs and the lab5090 independent Muse replay decoder
  were cancelled on the user's request; no new Muse listening audio is published.
- Six fresh Muse formal trains use the same LR grid as YuE2: 2e-5, 1e-4,
  3e-4 for each of SongEval and MuseCritic, 270 planned updates, beta=0.01,
  LoRA rank 8 and seed 42. SongEval uses wa66, MuseCritic iv96, priority 1023.
  Submission JSONs record real job IDs; queued is not training-complete.
  Native GRPO rollouts are retained, but independent inference and validation
  generation are disabled. Missing AUDIO, nonfinite audio or decoded audio
  under one second gets an explicit zero penalty and a separate invalid-output
  journal. Infrastructure errors remain fatal. This objective revision applies
  consistently to all six fresh Muse arms, not retroactively to older traces.

Every JSON snapshot records its real contiguous optimizer-step count,
checkpoint availability, source experiment SHA256, and audited reward/KL
trace. Muse failures are preserved, not filled with made-up final scores.

## Replay Integrity

YuE2 long inference uses max 1600 / min 200 semantic tokens; actual files are
approximately 64 seconds. Independent Muse inference is paused by request.
Fixed YuE2 baseline audio is shared across rewards, rates and checkpoints.
Public audio is MP3 only (192 kbps / 48 kHz, no loudness normalization), with
separate lossless-source and MP3 hashes. Lossless scoring originals remain
outside the repository; release assets must also be MP3. Waveforms use
fixed [-1, 1]; Mel spectra use 0–24 kHz and fixed [-80, 0] dB with black floor.

Audio filename suffixes encode SHA256 prefixes. Full hashes and byte counts
are in `asset-bindings.json`; the public registry binds prompts, seeds, actual
durations, checkpoints, audio and visuals rendered from the actual MP3.
Signal diagnostics in tables refer explicitly to pre-encoding originals.
Missing long-file scores are
explicitly unscored; short validation scores are not attributed to long audio.

## Interpretation

Training reward is a changing on-policy estimate, not held-out quality.
The right-hand curve is sampled training KL against the frozen reference,
with loss coefficient beta=0.01, not the old mock offline probe. SongEval and
MuseCritic share output key names but are different models and semantics;
their numerical scores are not comparable across backends. No listening-based
quality-improvement claim or official-test result is made here.

Muse's invalid/too-short rollouts are observed failures. They do not alone
establish reward hacking, and a higher reward does not establish perceptual
improvement. A corrected follow-up should give malformed outputs a bounded
explicit penalty and preserve sampling, conditioning and score-file bindings;
it must be reported as a separate run rather than editing these receipts.

YuE2 updates each two-song group only once. With dropout disabled, the
old/current ratio is one before that update; PPO ratio clipping is therefore
inactive in that gradient. The implementation computes raw-model action
log-probabilities while generation uses the native sampling transforms.
Consequently, the plotted KL is the logged sampled regularizer, not an exact
full-policy KL measurement. Audited receipts establish what ran and what was
scored; they do not certify an unbiased policy-gradient estimator or eliminate
these methodological limitations.
