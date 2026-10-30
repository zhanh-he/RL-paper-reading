# Real benchmark pilot · 2026-10-30

All numbers below are our measurements on declared subsets, not author-reported full-dataset results. Inference and adaptation completed on one RTX 5090. A duplicate pending scheduler job was cancelled after local GPU completion; no queued job is counted as a result.

## Results

| System / condition | Cohort | Independent metric | Result |
| --- | --- | --- | --- |
| Frozen SheetSage2, paper preset | 10 real GTZAN recordings | Beat / downbeat F1, 70 ms | 86.23% / 77.98% |
| Frozen DuET | 8 complete ASAP test works | MUSTER Mean ER, lower better | 4.504% |
| DuET duration-head SFT, 50 updates | Same 8 works | MUSTER Mean ER | 4.572% (+0.068 pp) |
| DuET note-action GRPO, 50 rollout batches / 40 updates | Same 8 works | MUSTER Mean ER | 4.516% (+0.012 pp) |

**No complete-score improvement is demonstrated.** SFT validation duration CE falls from 7.849 to 7.540 and exact-duration accuracy changes from 44.75% to 44.98% (one additional correct label out of 438), while test complete-score error increases. GRPO validation CE is 7.881 and duration accuracy is also 44.98%. Neither the small token gain nor the GRPO/SFT ordering proves generalization or superiority. This is one seed, a small nonrepresentative subset and unequal effective update counts.

SheetSage2 has a marked jazz failure: beat F1 17.48%, downbeat F1 0.00%. Keep this track in the mean. ASAP includes positive, negative and unchanged work-level differences; see `summary.json` and the expandable live table. The two visualized ASAP works were selected post hoc to illustrate an SFT improvement and deterioration, not to estimate average quality. The first four measures may not contain every changed event; full MusicXML is downloadable.

## Frozen protocols and provenance

- **GTZAN:** ten genres, first lexicographic available annotated file per genre, fixed before inference. Audio source [GTZAN mini](https://github.com/TempoBeatDownbeat/gtzan_mini) commit `a61439e86c13037011fde8e0f0743ec55c50bce3`; references [Beat This! v1.0](https://github.com/CPJKU/beat_this_annotations/tree/v1.0/gtzan) commit `c3c47fd37d3074d9f8119f18bbf460f909609f22`. Follow [SheetSage2 Appendix D.1](https://arxiv.org/html/2610.05336v1): discard reference and predicted events before 5 s, 70 ms F1, recording macro mean. This is not the paper's full 999/993-track cohort. GTZAN supplies no complete score ground truth here; only the rhythm component is scored.
- **SheetSage2:** [official bundle](https://huggingface.co/m-a-p/SheetSage2) revision `b83c3d79a6d98bcffaf8f9ed112c90aa887887bc`; MERT-v2-FullSong revision `d8ba1c745e733b3908ce6ad16ebeb17ac7600a42`; official `transcribe`, `preset="paper"`, BF16 on RTX 5090. Successful per-track times exclude model loading and are not cross-system latency comparisons. The first attempt failed on missing `pretty_midi`; a separate failure summary is retained, dependencies repaired in an isolated environment, all ten rerun.
- **ASAP:** [author-released dataset fork](https://github.com/TimFelixBeyer/asap-dataset/tree/8cba199e15931975542010a7ea2ff94a6fc9cbee), pinned commit `8cba199e15931975542010a7ea2ff94a6fc9cbee`; [ACPAS real metadata](https://cheriell.github.io/research/ACPAS_dataset/metadata_R.csv) supplies the split. SHA256-sort candidate performance paths, choose one performance per work: 16 train / 4 validation / 8 test. `asap-manifest.json` records every downloaded score, performance and annotation hash. The subset is work-disjoint for this adaptation but released-checkpoint pretraining overlap was not independently audited; do not call it eight unseen works for the base model.
- **DuET:** [official code](https://github.com/wei-zeng98/DuET) commit `6d2276c6d3b289584ffe4f602bc0fe1e4740f643`; [Wayne42/DuET](https://huggingface.co/Wayne42/DuET), epoch 83, weights SHA256 `89e07104358ca55ed3c8048c3d83b224c28e4ee0aa59e76a0ab0a9189ad114e0`. The official MIDI-only inference path is used; no audio transcription is implied. The architecture preserves retained input pitches and has a different error profile from free-form Audio-to-Score.
- **Independent scorer:** [MUSTER author fork](https://github.com/TimFelixBeyer/amtevaluation.github.io) commit `4ef1165edda1f719068c4c699bd8ab2076e4d7ec`, full MusicXML pairs, `clean_scores=False`, no added manual time/octave correction. MUSTER still uses its own internal alignment. All 8/8 completed in every arm; macro-average the returned per-work error percentages. The published six components are PitchER, MissRate, ExtraRate, OnsetER, OffsetER and MeanER; raw auxiliary voice/hand values are retained but are not a verified SATB evaluation.

This is not a reproduction of DuET's complete paper split or its main table. Do not compare our 4.504% directly against a full-paper result to infer an improvement.

## Small-data adaptation

Use the first official 32-second window, at most 128 notes per train/validation work. Warp performance onsets with paired beat annotations, match the same pitch with Hungarian assignment within 0.10 score seconds, and accept only a duration vocabulary value within one tick. One requested train work has fewer than 16 confident matches and is skipped: 15 usable works / 1,651 labels; validation is four works / 438 labels. These are our filtered alignment labels, not the author's original training labels.

Cache hidden states under frozen baseline-generated history, freeze encoder/decoder and all other heads. Optimize the 176,985-parameter duration classifier with AdamW (lr 1e-4, weight decay 0.01, gradient clip 1), seed 20261030. SFT performs 50 updates. The changed duration head is then used inside the original autoregressive full-piece decoder on all eight test works, not only the cached windows. Weight L2 change: SFT 0.362140; GRPO 0.106256.

GRPO samples eight singleton categorical duration actions per note/context, reward 1 for exact reference-duration class and 0 otherwise, group-normalized advantages, clipped policy ratio [0.8,1.2], exact forward KL to the frozen duration head (beta 0.01). One policy update per rollout batch means the initial likelihood ratio is one; this is an on-policy conditional-bandit/head adaptation, **not full autoregressive sequence-policy GRPO**. Fifty rollout batches produce 40 actual updates; ten all-zero-variance batches are skipped. It is exposure-matched to the SFT batches, not effective-update-matched. Raw batch reward/KL is conditional on different works and does not establish an upward learning trend.

No test labels enter training, early stopping or checkpoint selection. No model checkpoint is published. Complete-score error is an unoptimized downstream measurement; the duration reward cannot directly improve missing-note detection, onset errors or staff assignment. The negative result motivates sequence-level reward/independent-evaluation ablations; it does not demonstrate reward hacking or that a combined reward fixes it.

## Files and rights

`summary.json` is derived from the complete receipts. `duet-*-muster.json` retains each test work; `duet-baseline.json`, `duet-sft50.json`, `duet-grpo50.json` retain inference/adaptation summaries. `gtzan-manifest.json` and `asap-manifest.json` fix data identities. `sheet-dependency-failure.json` is a sanitized dependency-failure summary, not a scored run; raw machine tracebacks are excluded.

ASAP: Foscarin et al., *ASAP: a dataset of aligned scores and performances for piano transcription*, ISMIR 2020, [dataset](https://github.com/CPJKU/asap-dataset), CC BY-NC-SA 4.0 as supplied by the pinned fork. Public reference/performance MIDI and derived score excerpts are attributed in `platform/site/amt/demos/score-examples/README.md`. No GTZAN audio is republished; local lossless originals and model weights remain outside this repository.
