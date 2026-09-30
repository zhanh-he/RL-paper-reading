# Event-head GRPO pilot, 30 September 2026

This is the first ChoralStream experiment here that changes the **actual MIDI
event decoder**. The earlier frame-head GRPO improved its diagnostic frame
output but left decoded MIDI unchanged. This run updates the pitch, SATB voice,
onset and duration projection heads while the acoustic encoder, transformer and
frame head stay frozen. The group-relative reward, clipped policy ratio and
reference KL are implemented in [event_grpo.py](../../event_grpo.py); the
reference-backed reward is in [satb_reward.py](../../../../rewards/satb_reward.py).

## Protocol

- Same original ChoralStream checkpoint, seed 29 and fixed split for all arms.
- 16 original YouChorale-Pro training songs, 8 disjoint test songs; one 5.12 s
  segment at the same index from each song. The test songs are never sampled as
  rewards. This is not full-song evaluation.
- Four on-policy sampled MIDI rollouts per training segment; up to 64 tokens;
  two update epochs; AdamW learning rate `1e-5`; PPO ratio clip `0.2`;
  reference KL coefficient `0.01`. A zero-variance reward group skips the
  optimizer update and is counted separately from requested steps.
- Predeclared checkpoints at 100, 300 and 1000 steps for the combined arm.
  All seven single-reward arms use the same start, seed and split; their
  companion 300-step runs saved 100/300, while the 1000-step runs saved
  300/1000. Effective optimizer updates vary because zero-variance rollout
  groups cannot supply an advantage.
- A later [one-step rerun](step_000001/receipt.json) used the same starting
  checkpoint SHA-256, seed, 16/8-song split and optimizer settings, then
  stopped after one effective combined-reward update. Its pre-update
  aggregate is identical to the original baseline. This is a separate run,
  not an extracted checkpoint from the 1000-step run.
- Five displayed single-reward arms have 100-step checkpoints from their
  original 300-step runs. Their 300-step head files match the corresponding
  1000-step runs byte-for-byte. Separate one-step reruns used the same starting
  checkpoint, seed, split and optimizer settings. Anonymous aggregate receipts
  and the public-example replay are in [early-steps](early-steps/).
- Held-out MIDI is deterministically re-decoded. Frame F1 is computed on a
  10 ms grid; note onset matching uses 50 ms; complete-note matching also
  requires offset within `max(50 ms, 20% reference duration)`. We report
  pitch-only scores, SATB-track-aware scores, S/A/T/B per-part F1 and macro F1.

The public [aggregate receipt](aggregate.json) contains no song IDs, per-song
labels, audio or weights. [sanitize_event_receipts.py](../../sanitize_event_receipts.py)
verifies that all arms share the same checkpoint hash, split counts, seed and
baseline, then removes per-segment details. Original full receipts and event
head checkpoints remain on the research machine.

## Reading the result

At 1000 requested steps, held-out SATB per-voice macro F1 was:

| Reward | Effective updates | Frame | Onset | Complete note |
| --- | ---: | ---: | ---: | ---: |
| Frozen | 0 | 0.2917 | 0.1105 | 0.0269 |
| Combined | 1000 | **0.3857** | **0.1434** | **0.0419** |
| Onset only | 958 | 0.3666 | 0.1150 | 0.0317 |
| Onset + offset only | 606 | 0.3227 | 0.1080 | 0.0296 |
| Frame only | 1000 | 0.3425 | 0.1265 | 0.0144 |
| Coverage only | 1000 | 0.3077 | 0.0899 | 0.0213 |
| Continuity only | 192 | 0.3751 | 0.1027 | 0.0239 |
| Weak-voice only | 1 | 0.2783 | 0.1170 | 0.0340 |
| Precision only | 609 | 0.3459 | 0.1390 | 0.0396 |

The combined arm is best on all three reported held-out macro metrics for
this single seed and fixed 8-song split. It is evidence that the components
can complement each other, **not** proof that the weighted sum prevents all
reward hacking. Frame-only improves frame F1 while losing complete-note F1;
coverage and continuity do not preserve onset quality; weak-voice has almost
no reward support. The absolute complete-note F1 is still only 0.042.
Constructed counterexamples expose potential blind spots, not
optimizer-discovered exploits. More splits, seeds, full-song decoding and
human review are needed.

## Public listening replay

The [training curves](training-curves.json) contain only 25-step window means
from the six 1000-step training traces. Skipped updates are excluded; the
export includes the number of effective updates in each window. The combined
reward and KL panels show training-rollout values, not held-out F1. The five
single-reward plots come from separate ablation runs, not a decomposition of
the combined run. The export script is
[`export_training_curves.py`](../../export_training_curves.py); source song
identifiers and raw traces remain off the public site.

The [one-song receipt](public-synthetic-replay.json) and
[piano-roll notes](public-synthetic-notes.json) belong to an original 10.2 s
synthetic SATB example, not to the held-out test. The same audio input is
decoded with the frozen model, combined 1/100/300/1000-step heads, and the five
displayed single-reward 1/100/300/1000-step heads;
predicted notes are clipped to the input duration. Its MIDI files and ACE
Studio renders are on the [demo page](../../../../../../platform/site/demos/).
All ACE renders use the same Elirah / Emma / Julian / Mangus singer assignment,
the same project template and export settings. The original-synthesis input
and ACE-rendered reference are different audio realizations of the same
reference MIDI; objective scores come from the MIDI, not from ACE audio. The
weak-voice 300-step head has zero effective updates and its MIDI is
byte-identical to baseline, so the demo reuses the baseline render.
ACE singing tracks are monophonic, so overlapping notes within one predicted
voice are shortened to the next onset for listening only; the downloadable
MIDI and objective scores retain the original decoder output.

On this **one** example, pitch-only onset F1 is `0.133` at baseline, `0.103`
at 1, `0.195` at 100, `0.306` at 300 and `0.123` at 1000 steps. The 1000-step decline is a
listening case worth discussing, but not proof of overfitting or reward
hacking. Repeated seeds, more held-out songs and human listening ratings are
still needed. No Huawei training asset or proprietary recording appears here.
