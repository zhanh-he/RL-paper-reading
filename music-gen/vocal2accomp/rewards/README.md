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

[`audit_beat_v2_attacks.py`](audit_beat_v2_attacks.py) creates another **constructed, oracle-aligned** probe. It uses the scorer's eight reference beats from the six-second Emma vocal to place 60 ms sinusoidal clicks: original Madmom Beat-v2 F1 is `1.000` with STFT coverage `0.091`. Shifting the same clicks 350 ms later preserves STFT coverage `0.091` but drops Beat-v2 F1 to `0.000`. The [receipt](beat_v2_constructed_click_2026-09-30.json) and [listen-able demo](../../../platform/site/demos/vocal-lada.html) show that beat fit alone does not require any harmony or arrangement. Because the beat positions are taken directly from the metric's reference, this is a reward-function upper-bound attack, **not** evidence that LaDA generated or learned clicks.

The guarded Beat-v2 + coverage formula gives the public PCM click probe `-0.561`: its 40 ms RMS coverage is only `0.107`, so the beat-credit gate is `0.267`, and the loudness/flatness terms penalize it further. These component values are in the same [receipt](beat_v2_constructed_click_2026-09-30.json). This tests a known constructed attack, not the eventual trained guard arm or human preference.

Separately, a **real online LaDA rollout**, step 80 candidate 2 of the Beat-v2 arm, was captured with [`capture_beat_candidates.py`](../rl/grpo/capture_beat_candidates.py). The [WAV and independent-rescoring receipt](beat_v2_online_candidate_step80_2026-09-30.json) confirm six-second Beat-v2 F1 `0.308`, 40 ms RMS coverage `0.026`, original STFT coverage `0.000`, stereo RMS `0.0052`, and no clipping. The captured WAV's SHA-256 matches the receipt. The [demo](../../../platform/site/demos/vocal-lada.html?arm=beat_v2) provides raw playback and a clearly labeled +18 dB diagnostic copy. This is one stochastic training candidate, not a fixed checkpoint; without a matched frozen-policy rollout it cannot establish that training caused the sparse output.

Rescoring that **same saved PCM candidate** under the guarded Beat-v2 + coverage formula yields `-0.015`, with beat gate `0.066` and a `0.042` penalty for accompaniment more than 18 dB below the fixed vocal. That is a counterfactual reward comparison on one audio clip, not an output of the guarded policy or evidence that its training will be superior.

The same original scorer detected 15 vocal beats and 13 accompaniment beats in the fixed 12-second Beat-v2 step-150 replay, yet F1 was zero. [`render_beat_alignment.py`](render_beat_alignment.py) produced a [beat-time receipt](beat_v2_alignment_0_100_150_2026-09-30.json) and [timeline](../../../platform/site/demos/vocal-lada.html?arm=beat_v2) for baseline, step 100 and step 150. This measured checkpoint has high activity coverage, not silence; the issue is timing agreement with the fixed vocal according to the scorer. Listening is still needed to judge musical quality.

The separate [`beat_v2_worker.py`](beat_v2_worker.py) exposes the original Madmom Beat-v2 scorer to LaDA training through a persistent CPU process. It passed six- and twelve-second baseline preflights and completed a 300-step online arm. The coverage-only arm is complete: 23/100 rollout groups had equal reward for both candidates, including 22 where both reached the maximum 1.0; Beat-v2 had four tied pairs in its first 100 steps and 22/300 in total. Richness still lacks a validated implementation here; the spectral-band proxy must not be renamed to richness.

## Presentation interpretation

| Arm | Directly observed weakness | What remains unproven |
| --- | --- | --- |
| RMS coverage-only | Constructed drone/noise score 1.0; 23/100 training groups tie; fixed step 50 accompaniment is +3.4 dB over vocal before receding at 100 | Whether listeners prefer or reject step 50, or the model learned the constructed signals |
| Original Beat-v2-only | Oracle-aligned clicks score 1.0 with STFT coverage 0.091; one saved step-80 rollout has F1 0.308 with STFT coverage 0; fixed replay F1 rises at 100 then falls below baseline by 300 | Whether training caused the sparse rollout, or whether higher F1 implies better arrangement |
| Combined fast proxy | Constructed white-noise score falls to 0.148; fixed step 200 onset proxy rises while actual Beat-v2 F1 falls; step 300 accompaniment is +2.5 dB over vocal | Whether the combination generally prevents reward hacking; the band-occupancy proxy is constant across the seven fixed replays |

A next calibrated combination should use the **actual** Beat-v2 scorer, a non-saturated activity term, a validated musical-richness measure, and explicit vocal-relative loudness/peak guardrails. Each component needs its own held-out and listening checks; this is a design direction, not a claimed trained result.

The first partial implementation is [`beat_v2_coverage_guard_reward`](../rl/grpo/train_lada_band.py), which combines actual Beat-v2 and saturating 40 ms coverage with loudness/flatness guardrails. It is unit-tested and scored the oracle click attack at `-0.561`. Its matched GPU arm now has a 50-step [measured replay](../../../platform/site/demos/vocal-lada.html?arm=guarded): guarded reward rises `0.457 → 0.491` while original Beat-v2 F1 falls `0.296 → 0.216`. The full run is still in progress; this early checkpoint does not validate a quality improvement. Exact formula and command are in the [GRPO note](../rl/grpo/README.md).
