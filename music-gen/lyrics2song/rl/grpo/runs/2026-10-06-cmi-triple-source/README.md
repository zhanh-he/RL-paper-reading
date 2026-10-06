# CMI-Pref triple-source, text+lyrics GRPO rerun

**Submission:** 2026-10-06. Twelve two-step Gadi smokes and twelve dependent
100-step jobs were accepted by PBS at priority `-100`; no new scores are
verified in this receipt. A failed smoke leaves its own full run held rather
than publishing a result. The existing 234/59 text+lyrics runs remain a
separate historical protocol.

The source is the [frozen three-part CMI-Pref split](../../DATA-MODALITY-AUDIT.md):
240 train and 60 validation condition IDs, plus the sealed official three-part
test (125 votes / 121 conditions). Source manifest SHA256:
`b611333ef0abdeaf7a0473ea0beaa470673c3414f05b816a97904d2ade25b2e0`.
The models' **actual inputs are text+lyrics only**; the reference-audio field
is retained solely in the source manifest for future systems. Projected
manifest SHA256:
`b061588397d54177b25b678962caf756771498b927b9931542908c5e64a7e109`.
Muse's chat projection has 240 train / 60 valid rows and train/valid file
SHA256 values `37864c23487879ae21141270c0ebab046708d7f62ba66857c4c74b084c8969a8`
and `3c6ad89c4841041eb64f51536e544b5a925fd450e334f7742025a90f4f5be211`.
Its train prompts range from 127 to 1,384 model input tokens; all fit with a
512-token completion in the 2,048-token context. This does not guarantee a
complete song, and token-capped outputs must be reported as fragments.

| Model | Reward | LR values | Smoke IDs | Dependent 100-step IDs |
| --- | --- | --- | --- | --- |
| YuE2 | SongEval | 2e-5 / 1e-4 / 3e-4 | 180593664 / 666 / 668 | 180593665 / 667 / 669 |
| YuE2 | MuseCritic | 2e-5 / 1e-4 / 3e-4 | 180593670 / 672 / 674 | 180593671 / 673 / 675 |
| Muse | SongEval | 1e-6 / 3e-6 / 1e-5 | 180594381 / 383 / 385 | 180594382 / 384 / 386 |
| Muse | MuseCritic | 1e-6 / 3e-6 / 1e-5 | 180594387 / 389 / 392 | 180594388 / 391 / 393 |

YuE2 uses the custom two-rollout LoRA GRPO trainer with beta 0.01 sampled
frozen-base KL, logging every update and evaluating the fixed 60 validation
conditions at 0/1/25/50/100. The LR grid is compared only within the same
reward and compute stack. Muse uses ms-swift GRPO with two generations,
LoRA rank 8, beta 0.01, and checkpoints every 25 steps. Its SongEval plugin
scores exact decoded WAVs and records audio SHA256 bindings; its MuseCritic
plugin scores PCM24-FLAC files made from decoded WAVs. Both use the same
240-row train chat dataset and 60-row held-out dataset. Muse checkpoint
validation and audio-listening receipts are **not yet complete**; training
curves alone must not be presented as generalization evidence.
The initial Muse PBS pair set (`180594171` through `180594196`) was cancelled
while queued after a receipt-schema bug was found. It never ran. The table
above identifies the corrected resubmission.

Run code: [`gadi_yue2_triple_baseline.pbs`](../../gadi_yue2_triple_baseline.pbs),
[`gadi_muse_cmi_triple_baseline.pbs`](../../gadi_muse_cmi_triple_baseline.pbs),
[`muse_cmi_rewards.py`](../../muse_cmi_rewards.py).
Live Gadi outputs are under
`/g/data/wa66/hanyu/YuE2-posttrain/outputs/cmi_triple_textlyrics_*_20261006`
and
`/g/data/wa66/hanyu/vocal2accomp-muse/experiments/cmi_triple_textlyrics_20261006/outputs`.
