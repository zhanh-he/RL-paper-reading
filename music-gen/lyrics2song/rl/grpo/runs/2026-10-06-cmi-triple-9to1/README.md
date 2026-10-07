# CMI-Pref triple-source 9:1 formal protocol

This supersedes the queued 240/60 run; its 24 Gadi jobs were canceled while
queued or held. The older 234/59 and 240/60 experiments remain historical
records, not results for this protocol. No result is claimed from the new jobs
until logs, checkpoints, scored-audio hashes, and validation receipts pass.

## Frozen data

- Dataset: CMI-Pref, not CMI-Bench. Official train has 469 votes with nonempty
  text, reference-audio path, and lyrics, representing 325 unique conditions.
- Exclude any train vote whose normalized lyrics occur in official CMI-Pref
  test or WildSongBench. This removes 34 votes, leaving 435 votes / 300
  conditions. Group by normalized lyrics and order groups by SHA256. Allocate
  30 conditions to validation and 270 to train. The official same-modality
  test is sealed at 125 votes / 121 unique conditions.
- The [public split JSON](../../../../../../platform/site/demos/data/cmi-pref-triple-9to1-split.json)
  contains condition hashes and source prompt IDs, not lyrics or audio.
  Its SHA256 is `db769e27d75a70f2e819d6de771002652bb37e9adeb4a549fca66d9dfa30c616`.
- Private source manifest SHA256:
  `2f5f31219514ad479984e921a8dc6603680e275f31531778c7bb018da5df0136`.
  Text+lyrics projection SHA256:
  `c39d43e81793bee3f312c7028a0483b5da41068db4544acd258115043101e46f`.
  Muse train/valid JSONL SHA256:
  `ff437d751d62742d92daf7ac56a678010c52d6c22def5624a8324301e6699be6` /
  `ed5c5f36af48ef6ce81d15c7b15b429fb250ec4a52a04a43e55d96ec3cafb59d`.
- Rebuild with `prepare_cmi_triple_data.py --validation-conditions 30
  --manifest-name cmi-pref-triple-9to1-v2.json`, then
  `project_cmi_triple_baseline.py --validation-conditions 30`. Both source
  metadata files are SHA-pinned by the script. The public JSON supplies
  train/valid/test membership; the ignored local manifest contains the actual
  public prompt text needed by the model. The reference audio is **not**
  downloaded, decoded, or passed to either model in this run.

## Current formal jobs (2026-10-07 16:19 AWST)

The first Gadi smoke-gated submission is superseded. Its Muse jobs failed a
pre-training assertion because the script hashed an older dataset path; the
YuE2/MuseCritic smoke failed because torchaudio 2.10 required absent
TorchCodec for FLAC decoding. Both faults are corrected in the checked-in
scripts and deployed on Gadi. All remaining old queued/held jobs were
canceled. No old Gadi smoke score is a formal result.

The replacement jobs below start **full 270-step training directly** from a
fresh LoRA, with no smoke dependency. Gadi PBS confirmed priority `1023`
(maximum user-settable job priority), SongEval billed to wa66 and MuseCritic
to iv96. Training was queued at this snapshot, not yet verified complete.
YuE2 traverses all 270 training conditions once by index; Muse's ms-swift
loader controls sampling order, so 270 optimizer steps do not prove each
Muse condition was seen.

| Model | Reward | LR | Gadi full train | Fixed validation 0 / 270 |
| --- | --- | --- | --- | --- |
| YuE2 | SongEval | 2e-5 | 180732208 | integrated |
| YuE2 | SongEval | 1e-4 | 180732209 | integrated |
| YuE2 | SongEval | 3e-4 | 180732210 | integrated |
| YuE2 | MuseCritic | 2e-5 | 180732211 | integrated |
| YuE2 | MuseCritic | 1e-4 | 180732212 | integrated |
| YuE2 | MuseCritic | 3e-4 | 180732213 | integrated |
| Muse | SongEval | 1e-6 | 180732514 | 180733606 / 180733607 |
| Muse | SongEval | 3e-6 | 180732517 | 180733608 / 180733609 |
| Muse | SongEval | 1e-5 | 180732520 | 180733610 / 180733611 |
| Muse | MuseCritic | 1e-6 | 180732523 | 180733612 / 180733613 |
| Muse | MuseCritic | 3e-6 | 180732526 | 180733614 / 180733615 |
| Muse | MuseCritic | 1e-5 | 180732529 | 180733616 / 180733617 |

Muse validation jobs remain `afterok` dependent on their corresponding full
train, and also have priority `1023`. Kaya YuE2/SongEval LR1e-4 full job
`77858` is an FP16 cross-host replicate, not a fourth LR or a
dtype-controlled comparison. Its completed two-step smoke `77857` is not a
gate. Slurm accepted `Nice=0` (from 10000), then started the full job on
node `k018` at 2026-10-07 16:17 AWST. At 16:19, step-0 evaluation was in
progress; no formal reward comparison was ready.

At 16:27 AWST, idle lab5090 also started a full YuE2/SongEval LR2e-5 BF16
run using the same frozen dataset SHA256 and KL beta 0.01. Its launch script
is `run_lab5090_cmi_triple_9to1_songeval.sh`, remote output is
`/home/mengh/research/YuE2-posttrain/outputs/cmi_triple_9to1_songeval_lr2e-5_train_20261007_lab5090`,
and the initial supervisor PID was `3496974`. It is a separate cross-host
replicate of the Gadi 2e-5 arm, not an additional learning rate or a result
yet. It was launched only after `nvidia-smi` showed the 5090 idle; do not
interrupt other workloads to keep it running.

The first directly submitted Muse formal jobs and their dependent validation
jobs were still queued when an end-of-run checkpoint issue was found: 270 is
not divisible by `save_steps=25`, but the validator requires `checkpoint-270`.
Those queued jobs were canceled and replaced with the Muse IDs in the table
above, using `save_steps=30` so step 270 is saved. No started training was
discarded.

At 16:48 AWST, a prior Muse pilot showed ms-swift nests checkpoints under
`checkpoint/v0-<timestamp>/checkpoint-<step>`. The validation script's direct
`checkpoint/checkpoint-270` lookup was therefore wrong. It now requires
exactly one recursive match for the requested step, and the 12 not-yet-run
validation jobs were replaced by the IDs in the table. The six training job
IDs and their queue times were not changed.

### Verified early formal receipts, 2026-10-07 16:34 AWST

Kaya job `77858` remains running. The independent collector verified its
30-condition step-0 SongEval mean `3.413946`, all scored/archive audio SHA256
bindings, fixed indices/seeds, and the first training step's two scored WAVs.
All 30 step-0 generations report `truncated=true` under the 600-semantic-token
limit, so this is a short-generation baseline. Training step 1 has group mean
`2.651220` and sampled KL `0`; it uses a different training prompt and must
not be compared directly with the validation mean. No post-training validation
effect is yet claimed. The collector status was `partial-verified` (one
optimizer step, one validation checkpoint).

The first lab5090 attempt generated 30 step-0 WAVs but failed before writing
a receipt because the runner pointed SongEval at the YuE2 environment, which
lacked `librosa`. The correct pre-existing SongEval environment
`/home/mengh/miniconda3/envs/pytorch_env/bin/python` successfully scored all
30 generated WAVs in a manual preflight. The runner was corrected and
relaunched as supervisor PID `3497790`. The incomplete 30 WAVs are regenerated
from fixed seeds because their full rollout metadata was not archived; they
are not treated as a verified receipt.

At 16:39 AWST, the relaunched lab5090 run passed the same independent
collector: 30/30 step-0 scored/archive audio hashes matched, baseline
SongEval mean was `3.4636446667`, and optimizer step 1 had two verified
scored WAVs, finite gradient norm `0.050086`, and sampled KL `0` as expected
before the first update. Collector status was `partial-verified`. All 30
baseline generations were truncated under the 600-token limit. Kaya's
baseline `3.413946` was produced in FP16 versus lab5090 BF16, so the
cross-host baseline difference must not be read as a learning-rate effect.

At 16:46 AWST, lab5090's fixed 30-condition step-1 receipt was also
hash-verified: SongEval mean `3.4670353333` versus its own step-0
`3.4636446667` (delta `+0.003391`, too small to claim improvement).
Training steps 2/3 reported finite gradient norms and sampled KL
`0.0006825`/`0.0007817`; the run continued. These are wiring and early
trajectory checks, not final held-out or perceptual results.

At 16:42 AWST, a lab5090 on-host experiment queue (current PID `3498704`) was
started, waiting for the current 2e-5 process to exit. It will run
SongEval LR1e-4, then LR3e-4 in BF16 on the **same 270/30 split**, but only
after the preceding run passes `collect_formal_run.py` as
`verified-complete` and the GPU is continuously idle. The queue stops on a
failed receipt or existing output, and yields to any other workload occupying
the GPU. It is a remote experiment process, not a Codex scheduled task. Its
log is `outputs/cmi_triple_9to1_songeval_followups_20261007.log` on lab5090;
at this snapshot both follow-up arms are pending, not running or successful.

YuE2 evaluates 30 fixed validation conditions at steps 0/1/25/50/100/270,
logs training reward and sampled KL against the frozen base, and records
scored-audio SHA256. Muse training records reward/KL and checkpoints; its
separate fixed validation jobs cover steps 0 and 270. They remain unverified
until complete; intermediate Muse validation points are intentionally blank.
The official test is never used to pick LR or checkpoint. SongEval and
MuseCritic scores remain separate scales.

The YuE2 run uses at most 600 semantic tokens, so it is a short-generation
study, not a full-song benchmark. Neither reward is perceptual ground truth;
paired listening is required before claiming quality gains.

## Recovery snapshot: 2026-10-07

At 10:47 AWST, all 12 Gadi smoke jobs were still queued. Their 12 full runs
and 12 Muse validation jobs were waiting on resources or dependencies; no
Gadi formal result was available to collect. PBS reported insufficient GPU
capacity for the first queued YuE2 and Muse smoke jobs. Kaya job `77857`
(YuE2/SongEval, LR 1e-4, FP16) completed its **two-step smoke** with exit 0;
the dependent 270-step job `77858` was pending for priority. This snapshot
is not a claim that the full experiments have completed.

Kaya's smoke used two training and two validation conditions, not the full
270/30 split. The fixed two-condition validation SongEval means were 3.26271
at step 0, 3.24181 at step 1, and 3.05944 at step 2. All four scored
training WAVs and twelve validation audio files matched their recorded SHA256;
the three receipt means, indices/seeds, and checkpoint files were checked.
These values only establish that the pipeline runs and the receipts bind to
its audio. They are not formal effect estimates and are excluded from the
online results table.

Kaya smoke provenance: `experiment.json` SHA256
`f21c3763b1ea855d6156216621149e790729e21cb9140899b0f944ca39b5932e`;
`steps.jsonl` SHA256
`12e06f23614e6c7dbfc5b5f2760374cc4a8f7c9d109ed717c571669f8f0777b3`;
step 0/1/2 receipt SHA256 respectively
`ece826a9626b68e3ed52951667f5ca07db891bc0f86523a8f806dde70f837b09`,
`bc60c7bdf8af8d9194cf93320d604927d16a4fcbed492e4bdcd7aca007d803e4`,
and `07948b013bdb16913b8a4794ce75fc319c8630b839eedb80f655a6f1eda5fec8`.

### Queue reassessment, 2026-10-07 10:54 AWST

The user has access to both Gadi projects: wa66 (23.92 KSU available) and
iv96 (14.70 KSU available in 2026.q4). The original 36 Gadi jobs had all
been billed to wa66. The six MuseCritic arms and their validation chains
(18 pending jobs: YuE2 `180635253`-`180635258`, Muse `180635271`-
`180635276`, validation `180636767`-`180636772`) were changed in place to
iv96 with `qalter -P iv96`; SongEval remains on wa66. PBS confirmed the
changed project for each job, original queue time, dependencies, and
priority `-100`. No duplicate jobs were submitted. Both projects have ample
compute allocation, but they share the same gpuhopper GPU queue: PBS
reported zero currently available GPUs and no estimated start time for our
jobs. Project rebilling therefore does not establish a faster start.

Kaya's scheduler gave pending full job `77858` a provisional start of
2026-10-09 10:03 AWST (`squeue --start`, observed 2026-10-07 10:52 AWST).
This is a movable scheduling prediction, not a reservation or a completion
time. Its `Nice=10000`/priority 1 preserves other jobs' precedence.
