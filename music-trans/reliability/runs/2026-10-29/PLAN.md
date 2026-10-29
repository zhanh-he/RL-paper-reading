# Reliability and controllable-generation plan

1. E0/E0b: measured mathematical scorer controls. These are not model benchmarks.
2. E1: bounded, allocated SheetSage2 raw-audio smoke and perturbation audit.
3. E2: real singing evaluation after audio permission, note-label quality and
   song-disjoint splits are frozen. No invented real-song cohort or results.
4. E3: compare SOME-native melody-audio, SheetSage2 vocal-only MIDI and oracle
   monophonic MIDI in the same frozen YingMusic V1 generator. Hold lyrics,
   timbre reference/transcript, seeds, NFE, CFG, pitch shift and duration fixed.
   Start with 8 permission-cleared monophonic excerpts x 3 arms x 3 seeds (72
   outputs), not a choral-generation claim. Compare extraction before generation.

YingMusic-Singer (arXiv:2512.04779, 2025-12-04) jointly learns a melody extractor
supervised by the frozen SOME teacher, uses CKA alignment, and applies WER/F0
correlation rewards in Flow-GRPO. F0 Pearson is a reward/evaluation component;
this does not prove its generator is conditioned only on a raw scalar F0 curve.
The current V1 release and the later YingMusic-Singer-Plus paper are distinct
versions, not silently interchangeable replications of the 2025 paper.

## Released V1 interface: a stronger and narrower comparison

Pinned code: `6d7c453781cdaf286227ba7a51e79d57b19fad71`.
`extract_melody` uses RMVPE F0 for a voiced mask, then SOME returns framewise
MIDI-note values. The MIDI-file route also produces scalar note values, and
`sample` nearest-resamples, rounds and casts them to integer note indices.
Thus the released comparison is primarily an extractor/pipeline comparison
through a common discrete-note condition, NOT continuous-F0 vs full-score.

The MIDI parser is 100Hz; native audio features use 44.1kHz / 512-hop mel frames.
The wrapper concatenates reference and target melody before global resampling.
Verify reference/target boundary alignment with a click-note control before
interpreting differences. Common data type does not guarantee identical timing.
Report official-wrapper and explicitly timeline-normalized adapter experiments
separately; never silently patch one arm.

E0c executes only the official pinned MIDI parser on original one-second notes.
Check overlapping tracks in both orders. Full lead-sheet MIDI must not be used
as if it were polyphonic conditioning: overlapping notes are assigned into one
scalar array; track selection must be explicit. This port cannot test the value
of chord/voice/notation information that it discards.

Use identical input information for extractor comparison (same isolated vocal
for SOME and SheetSage2). Mixture vs isolated vocal is a separate front-end arm.
Preserve leading/trailing rests with a declared duration adapter, reject
overlapping lead notes, and log all failures. Oracle is an annotation-reference
condition, not a guaranteed upper bound on perceptual quality.

Gate: finish E1 load/runtime audit, verify V1 time grid, secure weights/audio
terms, and freeze the 8 excerpts before GPU generation. No singing generator
has been run in E0/E0b/E0c.

Comparing complete YingMusic and YuE2 outputs can be an end-to-end system study,
but is not a causal test of F0 vs score conditioning: generators, training data,
lyrics handling and output scope differ. Singing-only output must not compete
against full-song output with a whole-mixture quality score.

Evaluation should include absolute cents error, voicing precision/recall, octave
errors, strict note timing, lyric adherence, timbre, quality and inference budget.
Pearson alone is invariant to positive affine frequency transformations; DTW-only
scores can additionally hide duration errors. Fixed-timeline and aligned shape
scores must be separate. Use an independent evaluator or manual subset for audit.

Primary sources checked 2026-10-29:
- https://arxiv.org/html/2512.04779v1
- https://github.com/GiantAILab/YingMusic-Singer
- https://arxiv.org/html/2610.05336v1
- https://github.com/ASLP-lab/YingMusic-Singer-Plus
