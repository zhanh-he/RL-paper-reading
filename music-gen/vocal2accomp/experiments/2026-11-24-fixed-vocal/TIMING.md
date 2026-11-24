# Timing evidence for the fixed-vocal pilot

The timing tool runs Beat This with the CASM no-SMC decoder on the candidate audio at 50 Hz. It scores the isolated backing when the model supplies one. A mix-only engine is scored on its full mix, and a separated backing would need its own `backing_proxy` report. Each report binds the scored audio and reference by SHA-256.

The default pairing tolerance is 150 ms. Raw matching assigns proposed downbeats to reference times one to one, in order, maximizing the number of matches and then minimizing total absolute error. Positive lag means the generated event is later than its reference. The raw median uses only the raw matches. A separate offset search tests a single translation up to ±750 ms, using reference events outside the annotated uncertainty spans and detector events whose activation is at least 0.25. It requires at least three pairs and at least 45% coverage of eligible reference events. If the search has sufficient evidence, the corrected report includes each pair's original lag and residual. The audio stays in its original timeline.

The event confidence is the sigmoid of the backbone activation at the event frame. It is uncalibrated. A 100 ms RMS envelope records blanks of at least 350 ms below the greater of −55 dBFS and 45 dB below the candidate's strongest envelope frame. The comparison excludes events in those blanks and retains the decoder proposals and reason in the evidence. An unpaired reference, an extra generated downbeat, a signal below the gate and an uncertain reference position have distinct fields. The section boundary table reports the nearest detected downbeat as a timing observation. The boundary label remains the reference editor's judgment.

A provisional vocal navigation grid supports listening and inspection. The discrepancies remain provisional pending human verification. A candidate with an immediate audio onset can still have a later first proposed downbeat. Technical timing observations and listener preference require their own evidence.

## Run in the existing detector environment

Point `CASM_BUNDLE` at a locally provisioned CASM MIREX inference bundle containing the `mirex_pipeline` adapters, Beat This source and checkpoint, and CASM source and configuration. The pilot used Python 3.11, PyTorch and torchaudio 2.5.1, NumPy 1.26.4, SciPy 1.14.1, soundfile 0.13.1 and soxr 1.1.0 on CPU. The checkpoint, decoder configuration and adapter hashes go in the full internal evidence. The optional public JSON omits private detector provenance and vocal lyrics.

```sh
python evaluate_timing.py --bundle "$CASM_BUNDLE" \
  --audio /path/to/backing.wav --reference /path/to/reference.json \
  --kind backing --output /private/work/candidate-timing.json \
  --public-output /tmp/candidate-timing.public.json

python timing_controls.py --bundle "$CASM_BUNDLE" \
  --work-dir /private/work/timing-controls \
  --public-summary /tmp/timing-controls.public.json
```

A reference has `duration_s`, ordered `bars` with `bar`, `time_s` and `section`, and `uncertain_intervals` with `start_s`, `end_s` and `reason`. Optional `sections` records with `name` and `start_s` describe transitions independent of the bar grid. The repository contains the control generator. Rendered PCM controls and lossless inference files stay outside the repository.
