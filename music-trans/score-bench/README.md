# Paper-led AMT score benchmarks

[Live experiments](https://zhanh-he.github.io/RL-paper-reading/amt/demos/) · [14-system model table](https://zhanh-he.github.io/RL-paper-reading/amt/demos/#models) · [2026-10-30 results](runs/2026-10-30/README.md)

This track uses real benchmark recordings and performance MIDI, not procedural smoke inputs. Audio-to-Score, performance-MIDI-to-Score and engraving are different tasks; a beat F1, note F1 and score error must not share a ranking.

## This round

1. **SheetSage2 / GTZAN:** frozen official inference on ten recordings, one per genre. Follow Appendix D.1 for the rhythm component: exclude events before 5 seconds, match within 70 ms, macro-average recording F1. This does not evaluate complete lead-sheet transcription.
2. **DuET / ASAP:** download real performance MIDI, score MIDI, MusicXML and beat annotations. Freeze the selected ACPAS work groups before inference; evaluate all eight complete test works with the released MUSTER wrapper.
3. **Small-data SFT:** 15 usable train works / 1,651 confidence-filtered note labels, four validation works / 438 labels; update only the 176,985-parameter duration head for 50 steps.
4. **Note-action GRPO:** same cached contexts and labels; eight categorical duration samples per note, binary exact-duration reward, forward KL to the frozen head with beta 0.01. Fifty rollout batches yield 40 optimizer updates; ten batches have no reward variance. This is not full autoregressive sequence-policy GRPO and is not update-budget-matched to SFT.

Both adapted heads are evaluated through the original full-piece DuET decoder on the same eight test works. No test metric selects a checkpoint. One seed and a small subset support a feasibility/negative-result report, not a general SOTA or RL-effectiveness claim. The released checkpoint's original training overlap is unverified.

## Reproduction

Prepare isolated environments following the pinned official releases. Do not install these conflicting model dependencies into an existing training environment. Scripts accept external data/output directories; no model weights or lossless audio belong in this repository.

```bash
python prepare_asap.py --output "$ASAP_DATA"
python run_duet.py --data "$ASAP_DATA" --weights "$DUET_WEIGHTS" --output "$ASAP_DATA/duet-baseline"
python evaluate_muster.py --data "$ASAP_DATA" --predictions "$ASAP_DATA/duet-baseline"
python finetune_duet_duration.py --method sft --steps 50 --data "$ASAP_DATA" --weights "$DUET_WEIGHTS" --output "$ASAP_DATA/duet-duration-sft50"
python finetune_duet_duration.py --method grpo --steps 50 --data "$ASAP_DATA" --weights "$DUET_WEIGHTS" --output "$ASAP_DATA/duet-duration-grpo50"
python evaluate_muster.py --data "$ASAP_DATA" --predictions "$ASAP_DATA/duet-duration-sft50"
python evaluate_muster.py --data "$ASAP_DATA" --predictions "$ASAP_DATA/duet-duration-grpo50"
```

`prepare_gtzan.py --help` describes the pinned audio/annotation input trees. `run_sheet_gtzan.py` runs the official local Hugging Face bundles with the paper preset. `export_score_demo.py` exports the selected real MusicXML pairs. `publish_results.mjs` derives public tables from complete-cohort receipts; it rejects incomplete cohorts rather than silently averaging successes.

The primary-source model/dataset/reward map lives in `platform/data/amt/score-study.json`. This is an AMT **experiment** map, not a second literature catalog. Historical synthetic diagnostics remain separate and are not evidence of benchmark quality.

## Next controlled experiments

- Run MIDI2ScoreTransformer on the identical work manifest, with its required custom music21 toolchain; PM2S is a separate legacy-environment baseline.
- For complete Audio-to-Score evaluation use paired score data (ASAP piano / approved SheetSage-A2S / Quartets), not GTZAN beat labels as a score surrogate. Piano-A2S and MusicTime-A2S are runnable-release candidates; Rubato and TUTTI have unverified/unreleased checkpoint limitations in the checked resources.
- Compare SFT, frozen-policy candidate reranking and full-sequence GRPO at matched **effective** updates, rollout tokens and decode budget. Measure note content, timing, parse/bar validity and an unoptimized score metric separately.
- Ablate duration/onset/content/structure rewards and their combinations. A valid but empty/simple score can exploit structure reward; duration correctness does not guarantee note retention or voice/staff correctness. Combined rewards preventing hacking is a hypothesis, not a result of this pilot.
