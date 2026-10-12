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
- Muse fixed validation originally failed before generation because a deployed
  helper module was missing. The module was restored and 0/270 fixed-valid
  evaluations were resubmitted on Gadi iv96 (181026660–181026662). Training
  completion must not be confused with validation completion.

Every JSON snapshot records its real contiguous optimizer-step count,
checkpoint availability, source experiment SHA256, and audited reward/KL
trace. Muse failures are preserved, not filled with made-up final scores.

## Replay Integrity

YuE2 long inference uses max 1600 / min 200 semantic tokens; actual files are
approximately 64 seconds. Muse uses one continuous 1600-completion-token
trajectory and the official MuCodec overlapping decoder; actual length is
determined by AUDIO token count. Fixed per-model baseline audio is shared
across learning rates, rewards and checkpoints. Original FLAC lives in the
GitHub release to keep GitHub Pages below its site-size limit. Waveforms use
fixed [-1, 1]; Mel spectra use 0–24 kHz and fixed [-80, 0] dB with black floor.

Audio filename suffixes encode SHA256 prefixes. Full hashes and byte counts
are in `asset-bindings.json`; the public registry binds prompts, seeds, actual
durations, checkpoints, audio and visuals. Missing long-file scores are
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
