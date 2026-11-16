# SheetSage2 raw-audio audit

The frozen inference audit uses 12 procedural harmonic probes, each tested as
original, with 500 ms leading silence, at −6 dB gain and with 30 dB SNR noise.
These four conditions share the same 12 references. The evaluation covers
synthetic harmonic audio. The pinned model runs the paper preset with a 300-second window.

## Recovered on 2026-11-12

The archived inference completed all 48 attempts. Each condition has 12
measured probes and 12 nonempty ABC outputs. The pitch lists and counts used
for scoring match the corresponding melody MIDI exports in all 48 cases.
The worker requested BF16 on a V100-32GB. Effective kernel precision was unverified.
Wall time inside the worker was 1,097.7 seconds and peak GPU allocation was 10.42 GiB.

| Condition | Probes | Pitch-class onset F1 | Absolute onset F1 | Absolute onset + offset F1 |
| --- | ---: | ---: | ---: | ---: |
| Original | 12 | 100.0% | 100.0% | 22.2% |
| 500 ms leading silence | 12 | 100.0% | 100.0% | 22.2% |
| −6 dB gain | 12 | 100.0% | 100.0% | 22.2% |
| 30 dB SNR noise | 12 | 100.0% | 75.0% | 11.1% |

The recovery used the archived melody predictions and references with
mir_eval 0.8.2. The onset tolerance is 50 ms and the pitch tolerance is
50 cents. Offset matching uses 20% of the reference note duration with a
50 ms floor. The silence condition shifts the reference by 500 ms before
scoring. Each value is the macro mean over the same 12 procedural harmonic
probes. [Aggregate and artifact bindings](sheetsage2-e1-summary.json).
