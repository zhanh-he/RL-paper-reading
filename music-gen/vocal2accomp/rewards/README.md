# Vocal-to-accompaniment rewards

Keep simple implementations directly in this folder, not in three one-file subfolders:

| Intended file | What it should measure | Independent failure check |
| --- | --- | --- |
| `beat.py` | Timing fit to the **fixed original vocal**; Beat v2 is the requested first arm, v5 a version control | Repetition, drift, human rhythmic fit |
| `coverage.py` | Accompaniment activity in appropriate vocal sections | Noise beds, uninterrupted droning, missing sections |
| `richness.py` | Useful, coordinated arrangement layers | Unrelated layers, noise and harmonic clashes |
| `combine.py` | Calibrated composition of the three after single-arm tests | Floor on quality/coverage; no one score compensates for a hard failure |

Do not create these `.py` files as empty stubs. Migrate the actual, tested vocal2accomp definitions with their versioned dependencies and unit tests. Each reward should accept an explicit fixed-vocal reference, generated accompaniment and metadata, and return a scalar plus diagnostic components. `combine.py` should call the three modules and record their raw and normalized values; it must not hide them behind only one aggregate number.

## Fast proxy stress test, 30 Sep 2026

[`audit_proxy_attacks.py`](audit_proxy_attacks.py) scores **constructed signals, not LaDA outputs** against the fixed six-second Emma vocal using the exact proxy function in the LaDA trainer. The measured [JSON receipt](audit_proxy_attacks_2026-09-30.json) is reproducible with that input and trainer. It separates a reward-design diagnostic from the actual GRPO replay.

| Constructed signal | RMS coverage-only | Combined proxy | Interpretation |
| --- | ---: | ---: | --- |
| Sustained 220 Hz tone | 1.000 | 0.528 | Coverage is fully hacked by a drone. |
| White noise | 1.000 | 0.148 | Flatness penalty suppresses this broadband attack. |
| Vocal-gated three-tone chord | 1.000 | 0.797 | The combination can still reward a trivial envelope follower. |

The combined proxy is therefore a partial guardrail, **not** a validated music-quality reward. The separate coverage-only GRPO arm is needed to test whether the model actually discovers these or other failures; synthetic attack scores alone cannot establish model reward hacking. Beat-v2 and perceptual richness are not optimized in this proxy run.

The separate [`beat_v2_worker.py`](beat_v2_worker.py) exposes the original Madmom Beat-v2 scorer to LaDA training through a persistent CPU process. It has passed six- and twelve-second baseline preflights but has not yet produced an online Beat-v2 checkpoint. Richness still lacks a validated implementation here; the spectral-band proxy must not be renamed to richness.
