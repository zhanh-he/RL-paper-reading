# SongEval waveform perturbation audit

**Status:** measured reward diagnostic, not GRPO. We scored 47 evaluation-only
music excerpts with the [official SongEval implementation](https://github.com/ASLP-lab/SongEval).
No source audio, source IDs or individual song scores are redistributed. The
instrument is `songeval_audit.py`; `summarize_songeval_audit.py` creates the
public [aggregate receipt](perturbation_47clips.json).

For each clip, normalize mono peak to 0.5. Compare safe gain x1.6; hard clip
after x3.2; the same clipped signal RMS-matched to safe gain; and a 6 kHz
bandlimited/resampled control. SongEval's own inference path loads mono audio at
24 kHz and uses MuQ hidden state 6.

| Paired comparison | SongEval five-score mean delta | 95% clip-bootstrap interval | Positive clips |
| --- | ---: | ---: | ---: |
| Safe gain vs normalized reference | +0.0430 | [+0.0347, +0.0510] | 42/47 |
| Full-scale clip vs normalized reference | +0.0951 | [+0.0745, +0.1149] | 42/47 |
| RMS-matched hard clip vs safe gain | -0.0020 | [-0.0057, +0.0015] | 22/47 |
| 6 kHz bandlimit/upsample vs reference | -0.1828 | [-0.2341, -0.1288] | 6/47 |

The clean-loudness effect is clear. Clipping is not consistently preferred
when loudness is controlled; `Clarity` falls by 0.0059 on average in the
RMS-matched comparison. This does **not** establish how a long GRPO run behaves:
the policy could discover other reward shortcuts. Spectral energy up to 22.05
kHz in a 44.1 kHz output does not prove the reward saw that band, because
SongEval resamples to 24 kHz before scoring.

Bootstrap resamples clips, not training seeds; these excerpts are not a random
sample of all generated music. The independent model-update pilots live under
`../../rl/grpo/runs/`.
