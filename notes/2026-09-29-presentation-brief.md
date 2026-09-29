# Presentation brief: small-data music post-training

## Lead with the measured result

**Claim:** On one weak ChoralStream frame head, GRPO with 96 source-disjoint training songs raised held-out 5.12-second frame F1 from `0.1378` to `0.3033` (`+0.1656`, paired-song bootstrap 95% CI `[+0.1330, +0.2038]`, 30 songs, seed 29). Two exact-configuration repeats reached `0.3130` and `0.3134`. This is a small-data **head-level** result, not full SATB MIDI transcription. Weighted BCE on the same songs and update count reached `0.4541` frame F1. GRPO decreased onset and offset F1, so the interesting point is *reward design and task alignment*, not “GRPO always wins.” [Full protocol and receipts](../music-trans/choral-singing/rl/grpo/runs/2026-09-29-frame-head/README.md).

## Suggested four-screen flow

1. [Overview](https://zhanh-he.github.io/RL-paper-reading/demos/#overview): three statuses; state plainly that only the choral head experiment is measured GRPO.
2. [Choral GRPO](https://zhanh-he.github.io/RL-paper-reading/demos/#choral): switch from frame to onset/offset. Explain why one-dimensional improvement is insufficient and why BCE is an essential control.
3. [MIDI rewards](https://zhanh-he.github.io/RL-paper-reading/demos/#rewards): choose `Swap Soprano / Alto` (pitch+onset F1 `1.000`, track-note F1 `0.335`), then `Fragment sustained` (active-voice F1 `1.000`, track-note F1 `0.192`). The four existing prediction/reference pairs have track-note F1 `0.047` on average, but are a convenience diagnostic, not a random benchmark.
4. [Vocal demo](https://zhanh-he.github.io/RL-paper-reading/demos/#vocal): play 16-second synthetic guide and ACE-Step 1.5 completion. Say “local inference works, not GRPO, not proven pure accompaniment.”

## The honest outside comparison

[MuScriptor's paper](https://arxiv.org/abs/2607.08168) reports `41.7 -> 48.2` Multi F1 using 300 human-verified RL songs after much larger pretraining. That is **published, not locally replicated**; its released gated final checkpoints are not a clean pre-RL baseline. [MuseCritic's paper](https://arxiv.org/abs/2608.11755) motivates the lyrics-to-song arm; our H200 job is a one-step smoke, and a queued job is not a result. Model-size and weak/strong-model response remain hypotheses requiring matched experiments.

## Likely questions

- **“Is the data really small?”** The post-training pilot uses 96 train songs (one 5.12 s segment each); the *base checkpoint* was pretrained elsewhere. Never conflate the two budgets.
- **“Why does onset get worse?”** The scalar Bernoulli F1 reward acts on sampled binary frames, while reported F1 uses a fixed probability threshold. Reward/decode mismatch plus class imbalance can shift one head in the wrong direction.
- **“Why not reward all four voices?”** Real SATB voices can rest. A blanket four-active reward creates hallucinated notes. Use track-aware matching and per-voice recall where references exist; use silence-aware coverage only as a guard.
- **“Can we deploy tomorrow?”** Static demo and ACE inference can be shown. MuScriptor requires gated access; AnyAccomp requires environment adaptation; MuseCritic GRPO remains queued until an adapter and rollout receipt appear.

## Rehearsal checklist

Run `npm --prefix platform run check`, open the [live demo](https://zhanh-he.github.io/RL-paper-reading/demos/), check both audio controls and the four tabs, then verify Gadi job `180122896.gadi-pbs` before speaking. If the job completes, add its receipt only after checking actual adapter and rollout files. Do not put private YouChorale audio, manifests or checkpoints into the public repository.
