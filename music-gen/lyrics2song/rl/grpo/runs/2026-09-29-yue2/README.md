# YuE2 SongEval GRPO one-step pilot

**Status:** on-policy engineering pilot, measured on lab5090. The open
[YuE2-3B](https://github.com/m-a-p/YuE2) model generated two 24-second capped
training fragments from one original English lyric/style prompt. Their
[SongEval](https://github.com/ASLP-lab/SongEval) five-score means supplied
group-relative advantages; one LoRA GRPO optimizer step updated attention
q/v projections. No proprietary model or data was used.

| Held-out prompt, seed 5101 | SongEval mean | Peak | Samples >= 0.999 |
| --- | ---: | ---: | ---: |
| Before | 3.7383 | 0.885 | 0% |
| After | 3.3011 | 0.792 | 0% |

The held-out prompt uses different original lyrics and style. An untouched
baseline replay with the same seed produced identical semantic tokens and
identical decoded waveform samples. After the update, 502/600 semantic tokens
changed. The model therefore changed, but this single held-out SongEval score
**decreased**. Neither output approached full-scale clipping.

The [training receipt](receipt.json), [replay verification](verification.json),
and [pilot script](../../yue2_songeval_pilot.py) make this a checkable result.
Listen to the before/after pair in the [experiment demo](../../../../../../platform/site/demos/index.html#lyrics).
Both generations hit the 600-token cap, so neither is a full song. One group,
two rollouts, one update and one held-out prompt cannot characterize long-run
reward hacking, music quality, or a population effect. The generated audio is
presented for noncommercial academic comparison under the model's terms.
