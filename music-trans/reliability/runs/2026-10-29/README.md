# AMT reliability: measured controls

E0: the checked-in `metric-controls.json` is the genuine local CPU result from
mir_eval 0.8.2. It evaluates one artificial sequence of 12 notes, not a transcription
model or a dataset average. The original scorer is retained at `metric_controls.py`.

E0b: `f0-controls.json` is a newly executed NumPy mathematical control: scaling
all positive F0 values by two retains Pearson r=1 but incurs 1200 cents of error.
It does not run RMVPE, YingMusic-Singer or any generator. Pearson can be negative;
it is not reported as a bounded transcription F1.

E0c: `ying-midi-interface.json` executes only the pinned upstream MIDI parser
on one-second original overlapping-note controls. It does not initialize or
run the singing model. Track order is varied while musical note content stays
identical; the test detects scalar overwriting, not generation quality.

Listening previews are newly rendered sine tones from these artificial note
arrays. They illustrate pitch/duration errors; they are not audio from the original
scoring process and must not be called scored audio. `preview-hashes.json` records
the render-WAV and public MP3 hashes separately. MP3 is 192 kbps / 48 kHz, without
loudness normalization. MIDI and sine previews are original diagnostic material.

Frozen model audits and planned controlled-generation comparisons are separate
records. No reward/KL training curve or optimizer update exists in these controls.
