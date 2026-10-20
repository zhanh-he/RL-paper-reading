# Verified Historical Recovery, 2026-10-20

These are historical SP/subhead controls, **not SongBench Mean7 training**.
The approved reward definition and current experiment matrix are in the
[priority plan](../../SONGBENCH_MEAN7_PRIORITY_PLAN_2026-10-20.md).

`verified-recovery.json` was emitted by the read-only CPU
[`audit_completed_runs.py`](../../audit_completed_runs.py) on lab5090. No GPU
generation, new scoring, checkpoint loading, training or scheduler submission
was performed. Six runs passed: YuE2 and DiffRhythm2 x richness/instrument/
arrangement. Each has270 contiguous updates,660 SHA-bound training/validation
audio records, final checkpoint/optimizer/RNG files and metadata, identity KL
at the first update, and30/30 conditions at0/50/100/270. Total3,960 audio
bindings were checked. These are bindings, not3,960 independent songs; baselines
are shared within each model. Checkpoint files were hashed, not GPU-reloaded.

The actual historical remote code hashes match repository revision `e0dfd36`
(new experiment sources are allowed to evolve without changing this receipt):

| File | SHA256 |
| --- | --- |
| `run_diff_miles.py` | `8455bd32a4fd1bc3707f30c31183272bc2485bd8ca6966f2744074af2a7c920b` |
| `diff_audio.py` | `c87716d09cefe4676aa7f274a8f3359f1b9e4070603bc2692b84c5b968806c66` |
| `rewards.py` | `32c9409ee7f34ce016c176858205c6031f531b874a8b824280f1860ba7d01bec` |
| `train_yue2.py` | `d8d7dd10f0525ec83562300cf715bb0ecb697984920a7791364ac7d8a274e78a` |

The SongBench head hash is validated in every row. The old receipts do not
separately pin a MuQ checkpoint hash; this is an explicit provenance gap to
close before new Mean7 training. Do not overstate cross-environment identity.

Mean7 is retrospectively calculated from all seven saved head scores. It was
not the historical objective. YuE2 uses beta0.01/rank4 q-v/short semantic-token
budgets, Diff beta0.05/rank8 attention+MLP/60s budgets with selected-block SDE
updates. These are not controlled equal-budget cross-model MOS comparisons.
The public summary calculates paired-condition bootstrap95% intervals
(10,000 draws, fixed seed20261020), not across-training-seed uncertainty; they
are uncorrected for multiple comparisons. All seven-head results and training
traces remain in the audit. No partial means, head-only failures or mock runs
are imported into this recovery.

`listening.json` binds20 CPU-transcoded MP3s to the actual scored originals:
five predetermined validation indices0/10/20/1/2 x shared Diff baseline plus
three270-step endpoints. The exporter does not infer, pad or normalize audio.
The repo holds MP3 only,192kbps/48kHz; lossless originals stay on the lab machine.
These are unmodified CMI-Pref conditions, not the separate count2/count5 probes.
Dataset attribution is retained in the online replay inputs.

All six newly submitted Muse/Heart jobs from the previous round failed before
training: `ModuleNotFoundError: No module named 'generate_objective'` during
scorer/container bootstrap. Their scorer logs were checked on Gadi and the job
IDs no longer appear in current/history qstat. An expired job ID alone would
not establish failure; the startup trace does. Do not label them queued or
recover invented training endpoints. No existing job was resumed or canceled
in this audit. Old held Gadi/Kaya evaluation jobs were left untouched.
