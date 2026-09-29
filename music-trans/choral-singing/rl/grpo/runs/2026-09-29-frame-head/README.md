# ChoralStream frame-head post-training pilot

**Status:** local measured result, 2026-09-29. This is not a full SATB MIDI
transcription GRPO result. The frozen ChoralStream encoder produces mel features;
only its onset/frame/offset Bernoulli head is updated. Its autoregressive note,
duration and SATB voice decoder is unchanged.

## Protocol

- Starting checkpoint: local ChoralStream `choral_stream_youchorale_framebce_20260320_1337/ckpt/best`, SHA-256 `15a07b1dd21f85dd2e69ddbc11b4ef82c585ab4e90f362b7033175e2694393ff`.
- Data: local YouChorale-Pro, 96 distinct unaugmented training songs, 28 validation songs and 30 test songs, source IDs disjoint across these splits. One fixed 5.12-second segment at segment index 2 per song. We do not publish the source audio or private manifests.
- GRPO: 8 Bernoulli rollouts per audio, group-normalized reward, 4 clipped PPO updates per group, 3 passes over training songs, 1,152 optimizer steps; learning rate `5e-4`, clip epsilon `0.2`, frozen-reference Bernoulli KL coefficient `0.01`, seed `29`. No validation-based checkpoint selection. Clipping activated on only about `0.0001%` of sampled frame decisions, so the observed behavior is mostly group-relative policy gradient rather than active clipping.
- Supervised control: same starting checkpoint, train songs, segment, 3 passes, 1,152 optimizer steps, and learning rate; weighted BCE on all three heads. Its positive-class weights are fitted on the training split and capped at 100. This is a useful continued-training control, not an identical information budget: BCE sees frame labels directly while GRPO sees a scalar F1 reward.
- Evaluation: pooled precision, recall and F1 from thresholded frame-head probabilities on the fixed segments. Bootstrap resamples the 30 test songs as paired units 10,000 times (seed `20260929`). These are short-clip frame scores, not full-song note F1 or human quality judgments.

## Test Results

| Arm | Onset F1 | Frame F1 | Offset F1 | Frame change vs frozen |
| --- | ---: | ---: | ---: | ---: |
| Frozen checkpoint | 0.0070 | 0.1378 | 0.0084 | 0 |
| GRPO onset only | 0.0063 | 0.1475 | 0.0056 | +0.0097 |
| GRPO frame only | 0.0052 | 0.3033 | 0.0063 | +0.1656 |
| GRPO offset only | 0.0041 | 0.1388 | 0.0072 | +0.0011 |
| GRPO 0.25 onset + 0.50 frame + 0.25 offset | 0.0052 | 0.2997 | 0.0065 | +0.1620 |
| Weighted BCE control | 0.0485 | 0.4541 | 0.0386 | +0.3164 |

Two exact-configuration frame-only repeats used seeds 30 and 31, yielding test
frame F1 `0.3130` and `0.3134` respectively; seed 29 yielded `0.3033`.
Across the three training seeds, the observed test F1 range is `0.3033–0.3134`
(mean `0.3099`), all above the common `0.1378` starting score. This is a
small three-seed robustness check on one checkpoint and the same test clips,
not generalization across model sizes or datasets.

Frame-only GRPO's paired 95% bootstrap interval for the frame F1 change is
`[+0.1330, +0.2038]`; BCE's is `[+0.2818, +0.3519]`. The frame-only GRPO
also increases false positives from 37,466 to 62,931 and reduces onset and
offset F1. A raw weighted sum of rewards does not fix that trade-off. Onset-only
and offset-only increase their sampled training rewards slightly, but their
thresholded held-out F1 drops. This is a reward/decode mismatch to investigate,
not evidence that those heads improved.

## Full-song MIDI check

We separately decoded all 30 held-out YouChorale-Pro songs with the checkpoint's
autoregressive MIDI decoder (beam 2, SATB presence threshold 0.5). The frame-head
GRPO update changes only `frame_prj`; the event decoder does not use it. A paired
song decode and parameter comparison confirmed that the MIDI output is identical
before and after. These numbers therefore describe the frozen MIDI baseline, not
a GRPO gain:

| Match criterion | Pitch only F1 | Pitch + SATB track F1 |
| --- | ---: | ---: |
| 16 ms frame grid | 0.5542 | 0.3670 |
| Pitch + onset within 50 ms | 0.1063 | 0.0661 |
| Pitch + onset within 50 ms + offset within max(50 ms, 20% reference duration) | 0.0234 | 0.0145 |

Counts are pooled over complete songs, with mir_eval note matching for the last
two rows. This makes explicit the usual distinction between frame-wise F1,
onset-constrained note F1, and onset+offset-constrained note F1. It is not a
single joint "frame+onset" score. The [evaluation script](../../evaluate_notes.py)
and [aggregate receipt](note_metrics_50ms.json) include precision, recall and
the matched checkpoint/test-set hashes; song IDs and source audio stay private.

## Reproduction

Run `frame_grpo.py` on a machine with the local ChoralStream code, its checkpoint
and a licensed YouChorale-Pro copy. Set the four path arguments. Use
`--arm onset|frame|offset|combined` for GRPO or `--algorithm bce` for the
continued-training control. `bootstrap_test.py` consumes saved head weights.
The JSON receipts in this folder are the directly emitted aggregate results;
private rollout traces, manifests and checkpoints stay on the lab machine.

**Presentation conclusion:** small-data post-training of this weak frame head
can change held-out transcription scores, but this pilot does not show GRPO
beating supervised tuning, nor does it establish an improvement in SATB MIDI
note accuracy. The next proper GRPO target is an autoregressive event model
with track-aware note F1 and independently evaluated full-song MIDI.
