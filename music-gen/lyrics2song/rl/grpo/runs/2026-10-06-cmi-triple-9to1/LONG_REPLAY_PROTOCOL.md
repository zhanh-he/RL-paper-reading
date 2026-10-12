# Long-Form Replay Protocol

The 270-step formal runs use the frozen CMI-Pref 270/30 split. Their reward
curves are **short-generation training and validation**, not 30- or 60-second
song results. YuE2 uses a 600-semantic-token cap (all audited 30-condition
validation generations reached that cap at about 24 seconds). Muse currently
uses 512 completion tokens and decodes about 10.24 seconds. Do not relabel
either curve as long-form quality.

## Paired Listening Set

- Fix validation conditions 0, 10, and 20 before listening. They are not the
  sealed 121-condition official test. Project both models from the same source
  condition to their documented text+lyrics input; never supply reference audio
  as conditioning to one model only.
- Within each model, use one frozen baseline for every reward and learning-rate
  arm. Keep condition, prompt text, lyrics, numeric seed, sampler settings,
  decoder settings, and output format fixed across baseline and checkpoints.
  Different model families are not expected to produce identical waveforms
  from the same numeric seed.
- Show baseline and checkpoint 25/50/100/270 when those checkpoints exist.
  Mark absent snapshots honestly. Keep the existing short-reward and sampled-KL
  curves separately labeled; do not compute a fake 60-second KL from audio.
- For YuE2, request up to 1600 semantic tokens with the existing 200-token
  minimum, approximately a 64-second cap from the observed 25-token/s rate.
  Report actual duration and truncation for every file. Under-30-second
  outputs are failures of the long-form target, not samples to discard, pad,
  stretch, or silently replace. The first paired render is baseline versus
  SongEval LR 3e-4 checkpoint 270 on lab5090.
- For Muse, request one continuous 1600-completion-token trajectory with
  `force_length=True` for both baseline and adapters. Decode with the official
  MuCodec `duration=40.96` window, not the short evaluation's `10.24` crop.
  MuCodec determines total length from AUDIO token count at 25 tokens/second
  and natively overlap-decodes longer sequences (1024-token window,
  768-token hop). The duration argument is a decoder window, not a target song
  duration. Count actual AUDIO tokens and report actual seconds; text tokens
  may make a 1600-completion-token output shorter than 64 seconds. Do not
  concatenate independent songs, repeat audio, or pad to meet the target.
  Muse checkpoints are saved at multiples of 30, unlike YuE2's 1/25/50/100;
  show their real optimizer steps and retain failed-arm partial checkpoints.

## Measurement And Presentation

- Archive the exact displayed PCM24 FLAC plus scorer input WAV (when used),
  SHA256, sample rate, actual duration, generation tokens, seed, checkpoint
  adapter hash, prompt condition ID, and peak/RMS/clipping metrics. Score only
  the exact audio file named in the receipt. Baseline must remain identical
  when switching optimizer step or reward arm within a model.
- Show 30-second and 60-second playback targets. Prefer an uncropped complete
  generation near 60 seconds. If it ends early, display the measured length.
  If it exceeds 60 seconds, retain the uncropped master and label any
  presentation excerpt as an excerpt.
- Treat SongEval/MuseCritic full-file scores as exploratory until their
  duration behavior is checked. Also score fixed beginning/middle/end windows
  with identical windowing across A/B, and report duration, clipping, loudness,
  silence, and continuity separately. Do not mix a short training reward with
  a long playback score in one curve or compare SongEval and MuseCritic scales.
- Choose a checkpoint using the frozen 30-condition validation only. Render
  the official test once after checkpoint selection, without choosing examples
  or parameters from test outcomes. The three validation replay examples are
  illustrations, not an estimate of test-set improvement.

The current YuE2 renderer is `render_yue2_long_replay.py`; its manifest and
per-audio receipts are written under the verified training run's
`long_replay/step_000270/` directory. No long-form audio is a verified demo
asset until those receipts and the listening check pass.
