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

## Submitted arms

All Gadi jobs were accepted at priority -100. Each two-step smoke is a
prerequisite for its distinct 270-step training job, which starts from a
fresh LoRA. The 270 planned YuE2 updates traverse all 270 training conditions
once by index. Muse's ms-swift loader controls sampling order; 270 optimizer
steps must not be described as proof that every Muse condition was seen.

| Model | Reward | LR | Smoke | Train |
| --- | --- | --- | --- | --- |
| YuE2 | SongEval | 2e-5 | 180635247 | 180635248 |
| YuE2 | SongEval | 1e-4 | 180635249 | 180635250 |
| YuE2 | SongEval | 3e-4 | 180635251 | 180635252 |
| YuE2 | MuseCritic | 2e-5 | 180635253 | 180635254 |
| YuE2 | MuseCritic | 1e-4 | 180635255 | 180635256 |
| YuE2 | MuseCritic | 3e-4 | 180635257 | 180635258 |
| Muse | SongEval | 1e-6 | 180635265 | 180635266 |
| Muse | SongEval | 3e-6 | 180635267 | 180635268 |
| Muse | SongEval | 1e-5 | 180635269 | 180635270 |
| Muse | MuseCritic | 1e-6 | 180635271 | 180635272 |
| Muse | MuseCritic | 3e-6 | 180635273 | 180635274 |
| Muse | MuseCritic | 1e-5 | 180635275 | 180635276 |

Kaya additionally accepted YuE2/SongEval LR1e-4 smoke `77857` and
dependent train `77858` at low priority. This is an FP16 cross-host replicate,
not a fourth LR or a dtype-controlled comparison. Its GPU partition is
capacity-constrained; submission is not completion.

YuE2 evaluates 30 fixed validation conditions at steps 0/1/25/50/100/270,
logs training reward and sampled KL against the frozen base, and records
scored-audio SHA256. Muse training records reward/KL and checkpoints; its
separate fixed validation jobs for steps 0 and 270 are submitted with
`afterok` dependencies on the corresponding full runs: SongEval 1e-6
`180636761/762`, 3e-6 `180636763/764`, 1e-5 `180636765/766`;
MuseCritic 1e-6 `180636767/768`, 3e-6 `180636769/770`, 1e-5
`180636771/772`. They remain unverified until complete; intermediate Muse
validation points are intentionally blank. The official test is never used to pick
LR or checkpoint. SongEval and MuseCritic scores remain separate scales.

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
