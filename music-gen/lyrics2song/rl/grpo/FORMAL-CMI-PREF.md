# Formal lyrics-to-song GRPO, 2026-10-05

This protocol supersedes the eight-handwritten-prompt pilot for any claim about
generalization. The old pilot remains available under **Mock Experiment** in
the [demo](https://zhanh-he.github.io/RL-paper-reading/demos/#lyrics).

## Data

`prepare_formal_data.py` reads pinned public metadata into ignored `datasets/`
and writes `datasets/formal-lyrics2song-v1.json`. The manifest SHA256 is
`c29217883289d3717c510d671927319d5e7acb70ad00443d2a3d9384653ac23b`.
Source revisions and file hashes are in
[`formal-data-summary.json`](../../../../platform/site/demos/formal-data-summary.json).
The separate [modality audit](DATA-MODALITY-AUDIT.md) explains why this is a
text+lyrics task, not a reference-audio task.

- [CMI-Pref](https://huggingface.co/datasets/HaiwenXia/cmi-pref): 3,527 official
  train votes and 500 official test votes. Exactly 414 train votes contain both
  lyrics and style without requiring reference audio. Remove 23 votes whose
  normalized lyrics occur in the official test; deduplicate 98 repeated
  style/lyrics votes. This leaves 293 independent conditions: 234 train and
  59 validation, split by stable SHA256 order. These are *not* all 3,527 rows.
- [WildSongBench](https://huggingface.co/datasets/m-a-p/WildSongBench): all 192
  public prompts are sealed as the external test. No WSB prompt is used for
  training, validation, learning-rate choice, or checkpoint choice.
- [SongEval](https://huggingface.co/datasets/ASLP-lab/SongEval): the audio
  evaluator, not a lyrics/style prompt corpus. Its published viewer shows
  1,000 public audio rows in a `train` split.
- [CMI-RewardBench](https://github.com/Haiwen-Xia/CMI-RewardBench): a
  reward-model evaluation benchmark, not a YuE2 generation prompt set.

The raw metadata, private generated audio and full manifest stay out of git.
Exact normalized-lyrics decontamination does not catch paraphrases; a human
review is required before a strong no-leakage claim. The three old hand-written
demo prompts are illustrative only and never enter formal validation curves.

## Objective and evidence

All arms start from the unmodified released YuE2-3B weights with a newly
initialized rank-4 LoRA (`q_proj`, `v_proj`); they do **not** start from the
one-step pilot adapter. One condition yields two on-policy songs per optimizer
step. The five-component reward is averaged, then group-normalized. The update
also penalizes sampled k3 KL against the frozen base with beta `0.01`. Both
training reward and sampled KL are logged for every optimizer update, and the
demo uses 10-step moving averages with raw traces. The two reward models have
different scales: never compare their absolute values.

The fixed validation set is all 59 conditions, seed `5101 + index`, evaluated
at steps 0/1/25/50/100. This is separate from the online training trace.
SongEval scores float WAV; both its exact scored WAV and the playback PCM24
FLAC are retained and hashed. MuseCritic scores the **same PCM24-FLAC bytes**
that are archived. Every training rollout likewise retains its exact scored
audio and SHA256. `collect_formal_run.py` refuses to publish a complete result
unless all 100 steps, 59 items per validation checkpoint, and audio hashes
verify. The baseline and post-training clips are short (600 semantic tokens);
they are not the official WildSongBench full-song generation protocol.

The current optimizer makes only one gradient update per two-rollout group;
therefore its old-policy ratio is one when evaluated and clipping has no
effective trust-region action at that instant. The KL penalty is the active
regularizer. The reward has no explicit lyric alignment, duration, peak, or
clipping term, so those remain guardrail measurements rather than guaranteed
improvements. Human listening is still needed before a quality claim.

## Submitted jobs

| Host | Arm | Smoke | Dependent 100-step job | Priority |
| --- | --- | --- | --- | --- |
| Gadi H100 | SongEval, LR 2e-5 | `180532793.gadi-pbs` | `180532807.gadi-pbs` | PBS `-p -100` |
| Kaya V100-32GB | SongEval, LR 1e-4, FP16 | `75964` | `75965` | Slurm `--nice=10000`, after CASM `75204` |
| Gadi H100 | MuseCritic PCM24, LR 2e-5 | `180533336.gadi-pbs` | `180533343.gadi-pbs` | PBS `-p -100` |

Each 100-step job has an
`afterok` dependency on its own two-step smoke. Kaya is intentionally running
a SongEval LR/FP16 feasibility arm: the official YuE2 pipeline loads BF16, so
its independent Kaya checkout explicitly loads FP16 instead. The smoke must
first verify numerical stability. Its difference from Gadi BF16 **cannot** be
attributed to LR alone. The official MuseCritic server also hard-codes BF16,
which the V100 does not support; no scorer-precision change is hidden behind
a comparable MuseCritic score. CASM beat work retains scheduling priority.
Kaya's smoke also has an `afterany:75204` dependency on the already queued
CASM continuation, so it cannot start ahead of that beat experiment. The
original general-`gpu` YuE2 queue pair was cancelled after the new pair was
accepted; no duplicate GPU jobs remain.

The source scripts are `gadi_yue2_formal.pbs`,
`gadi_yue2_musecritic_formal.pbs`, and `kaya_yue2_formal.sbatch`. The Gadi
outputs live in `/g/data/wa66/hanyu/YuE2-posttrain/outputs/cmi_pref_*_20261005`;
the Kaya outputs live in `/scratch/ems011/zhe/yue2-formal/outputs/`.
No Codex recurring task was created.

## Verified SongEval readout, 2026-10-06

Both SongEval arms finished with exit status 0. `collect_formal_run.py`
verified 100 contiguous optimizer steps, scored-audio SHA256 bindings, and
all 59 validation entries at each of steps 0/1/25/50/100. The public
demo contains their complete lyrics-free curves and validation receipts.

| Arm | Step 0 | Step 1 | Step 25 | Step 50 | Step 100 | Step 100 minus 0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gadi H100, BF16, 2e-5 | 3.417 | 3.429 | 3.454 | 3.448 | 3.432 | +0.015 |
| Kaya V100, FP16, 1e-4 | 3.469 | 3.465 | 3.440 | 3.459 | 3.484 | +0.015 |

These are fixed-prompt SongEval means, not human preference or lyric-alignment
results. Exploratory paired bootstrap 95% intervals for step 100 minus 0 are
[-0.088, 0.121] on Gadi and [-0.053, 0.081] on Kaya; both include zero.
All 59 validation samples at every checkpoint report semantic-token
truncation, so these are short clips. No audible quality improvement or
full-song generalization claim follows from this table. The FP16/BF16 change
prevents attributing a cross-host difference solely to learning rate.
MuseCritic remains queued and has no verified formal result in this readout.

## Final test gate

Choose one arm/checkpoint using the 59-condition validation set and human
listening before opening WildSongBench. Then generate baseline and selected
checkpoint for **all 192** WSB prompts with matched seeds, report the full
sample count, score distribution and signal guardrails, and listen to paired
examples. Because this run caps output at 600 semantic tokens, describe it as
an all-prompt *short-clip* WSB test; official full-song metrics require the
published inference and evaluation protocol. `evaluate_wildsongbench.py` is
staged on both clusters but its test job is deliberately **not** submitted
until a single checkpoint is selected on validation. It pins the WSB manifest
SHA, uses each row's first published AR seed for both baseline and candidate,
and hashes every exact scored audio file.
