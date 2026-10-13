# Lyrics2song Objective Test Evaluation

The objective tables distinguish **our measured final-step checkpoints** from
**the YuE2 authors' published WSB comparison**. Test evaluation was authorized on
2026-10-13. No new training is launched; YuE2-SongBench GRPO is reserved only.

## Cohorts

| Dataset | Unit | Generation and aggregation |
| --- | --- | --- |
| CMI-Pref official triple-source test | 121 conditions / 125 preference votes | Frozen public split IDs, two fixed seeds, all 242 candidates averaged. Input text + lyrics; reference audio reserved, not consumed. Preference recordings are not unique generation targets. |
| WildSongBench | All 192 prompts, 94 Chinese / 98 English | Two official seeds per prompt; four ASR passes per candidate; choose lower PER with official tie-breaking. Every other metric sees the same selected audio. |

Keep the existing 270 train / 30 validation memberships unchanged. Every completed
YuE2 formal arm is evaluated at **fixed final step 270** with LR 2e-5 / 1e-4 /
3e-4, under both SongEval and MuseCritic. Do not select checkpoint or learning
rate from either test set. Comparing all preregistered arms is not a license to
tune on the test results. Muse replacement arms still await training; old small-LR
or Mock runs cannot fill their rows. Muse baseline evaluation is not submitted yet.

## Metrics

The [official nine-column comparison](https://github.com/multimodal-art-projection/YuE/blob/main/docs/benchmarks.md)
includes SongBench mean and Musicality, SongEval mean and Musicality, AudioBox
PQ, MuQ-MuLan, AllMusicCaps, Q3O control and phoneme error rate (PER).

- [SongBench](https://github.com/Tencent/SongBench): Vocal, Instrumental, Melody,
  Structure, Arrangement, Mixing, Musicality; mean of seven, not SongEval mean.
- SongEval: five independent SongEval-head scores; mean and Musicality.
- [AudioBox](https://github.com/facebookresearch/audiobox-aesthetics): Production
  Quality (PQ). Store CE / CU / PC too, but headline comparison follows PQ.
- MuQ-MuLan / AllMusicCaps: audio similarity to requested **style text**, not lyrics.
- Q3O: prompt-control score; official WSB requires 80 GB GPU. Submitted to
  Gadi Hopper allocations, never substituted with a smaller model on the 5090.
- PER: corpus micro average, sum edits / sum reference phonemes, not mean of
  per-song percentages. Preserve incomplete lyrics and capped ASR outputs.

CMI-Pref's complete test cohort includes Vietnamese and Javanese lyrics. The
frozen Chinese/English G2P is not a validated multilingual reference: full-cohort
PER is `n/a`. Do not transliterate, drop foreign songs, or label all Latin script
as English to produce an apparently comparable number. A future language-audited
subset would need its own denominator and protocol. CMI Q3O is also `n/a` for
the fixed WSB metric: CMI has no corresponding frozen musical-attribute weights.
New Q3O annotations would constitute a separately versioned extension.

## Reproduction Boundary

WSB sources are pinned at revision
`e361b85a8d7365d3079fa9647641a87ce3c8c5b5`.
The evaluator archive SHA256 is
`bd550b36f1fc43cad3254a9897dd3a078e50c4eb9e8d4f295ac4b5e81d1fc4fd`.
The prompt manifest SHA256 is
`859e5225d319912efb64d587d1d5eb6cf0abc9fb162cc3d5f02d1688cff2edc0`.
The CMI test source SHA256 is
`824a92f33d04096e739fdc771d09bda125943baafae5c0eb5b6e9e267d3762e9`.
Membership is checked against the downloadable
[frozen split](../../../platform/site/demos/data/cmi-pref-triple-9to1-split.json).

Native YuE2 inference: full symbolic planning, CFG 1, unquantized bf16 AR,
FP32 benchmark `YuE2-Vae-legacy` revision
`b54118f0fc462f08999d1ec07e88817f4ee3f770`, 200–9000 semantic tokens,
32-step midpoint acoustic sampling, torch-eager backend. Both capped and
uncapped outputs stay in the cohort. This differs from training's 600-token cap
and the listening demo's 1600-token / modern-VAE setting.
LoRA is merged into the native bf16 AR/NAR model before evaluation; the original
adapter file SHA is recorded separately from the base weight identity.

The frozen WSB CLI intentionally rejects adapted generator weights and non-WSB
prompts. **Do not patch it to pass acceptance.** Our wrapper invokes its original
workers, verifies their frozen code/assets/environments and binds each score to
audio and adapter hashes. These are local checkpoint/CMI extensions, not exact
paper acceptance runs. SongEval and AudioBox are supplementary implementations;
store their runtime/config/model-state hashes and do not claim identical paper
preprocessing for these two score families.

The original SongBench eval environment is Torch 2.7.0+cu126, which lacks RTX
5090 sm_120 kernels: the first four scores failed while the eight PER receipts
completed. `score_songbench_blackwell.py` reuses the original SongBench model,
checkpoint, FP32 MuQ frontend and seed, but runs in the frozen metrics environment
with Torch 2.9.1+cu128. Its source/runtime hash is included in scoring identities.
The original code and lock files are not altered; the original failure log and
environment receipt remain archived privately. SongBench is therefore a **Blackwell runtime extension**,
not a frozen SongBench-environment acceptance run.

Unlike training reward, these objective evaluators consume fixed 48 kHz / 192
kbps MP3 encoding, following the WSB input contract. No loudness normalization.
Private FLAC generation originals and public scoring MP3s have separate hashes.
Neither restricted data nor private audio is copied to this repository.

## Execution And Recovery

`run_objective_queue.sh` uses a lock and checks for other GPU processes before
each chunk. It stops without interrupting another job. It runs all 192 / 121
conditions, not just a smoke sample; four-condition chunks are resumable
publication/checkpoint boundaries, not subsets used as final benchmark scores.
For the distributed campaign, `EVAL_END=4 SCORE_WSB=0` restricts the 5090 to
the first four conditions of each cohort; WSB ASR/Q3O scoring is delegated to
Gadi. The replacement controller uses `SCORE_CMI=0` to prepare immutable MP3
chunks for Kaya instead of occupying the generation GPU with quality scorers.
It preserves the three CMI arms already scored locally and lets the old
controller's in-flight child finish under the same lock. An actual CUDA OOM stopped the original official ASR worker at
SongEval-3e-4; its generated audio was retained, not regenerated or discarded.
An incomplete result directory requires inspection; no overwrite or random reseed.
Scorer failure stops the queue, retaining the affected audio and traceback.

`collect_objective.py` publishes denominators, hash receipts and measured
coverage. Headline numeric cells are withheld until their full metric cohort
is complete. `pending` / resource-pending / not-applicable remain distinct.
`official-wsb-results.json` is the unchanged official published JSON, with
snapshot date and source provenance, not locally generated results.

First complete run covers 7 YuE2 arms × (384 WSB + 242 CMI) = **4,382 full-song
candidates**, plus scoring. See [actual submission receipts](submissions-20261013.json).
Gadi wa66/iv96 each have six disjoint single-Hopper jobs; each owns the same
conditions across all seven model arms and includes Q3O. A thirteenth Hopper
job finishes the 5090's first four WSB conditions, reusing the five completed
selected-audio chunks and scoring the two remaining arms. The six existing
Muse training jobs are not cancelled, suspended or preempted. Both projects
share a physical queue, and the per-user six-running-job limit prevents a claim
of thirteen simultaneous Gadi GPUs. Submission priority is 1023.

Downloads and installation run in Gadi `copyq`: ordinary compute nodes cannot
reach Hugging Face. GPU jobs depend on successful preparation. Kaya has a
separate preparation and V100 quality-scoring job on five arms' already-selected
WSB MP3s. It checks cross-host numerical drift without changing candidate draws
or adding duplicate observations to the headline cohort. No BF16 native YuE2
generation, official BF16 ASR, or 80GB Q3O runs on Kaya's V100s.
This audit completed on `k020`: 20 scored MP3s, exit 0, maximum absolute
cross-host difference 0.0002432 across audited score dimensions. This is a
numerical compatibility check, not a full-cohort or perceptual improvement.
`kaya_cmi_quality.sbatch` scores new unselected CMI chunks prepared on the 5090;
these are production test observations, not additional draws. Each arm has a
separate input/output root and retains both seeds and generation receipts.
Two SongEval-3e-4 songs exceeded a 16GB V100 in full-song MuQ attention; the
replacement job requires a 32GB V100. No cropping or scoring math changes.
`scorer_cache.py` archives original failed receipts before explicit retries;
completed receipts remain cached and are not silently rescored.
The finite `dispatch_reserved_cmi.py` transfer stage handles exactly the three
remaining reserved chunks, streams MP3s directly between SSH hosts, submits one
native quality job per arm, and exits. It never collects/publishes future results,
creates schedules, starts training, or sends notifications. Its bounded execution
receipt is `/tmp/lyrics-cmi-quality-transfer-20261013.json` on this Mac.

Use repeat `collect_objective.py --runs ...` arguments for disjoint roots: Gadi
`first-four` for WSB, the 5090 reserved CMI root, and the twelve Gadi `shard_XX`
roots. Do not also include the 5090's original selected WSB chunks or Kaya's
numerical audit. Duplicate IDs and differing adapter/scorer hashes are rejected.
For the reserved CMI range, use 5090 results for baseline/SongEval-2e-5/
MuseCritic-2e-5, and the separate Kaya `cmi-quality/<arm>` roots for the other
four arms. Alternatively, `merge_returned_quality.py` verifies and recovers each
completed Kaya chunk into its original 5090 run, replacing only that arm's unscored
placeholder. After this recovery, collect the original 5090 prefix alone, not
both copies. Bound MP3s, source FLACs, candidates, seeds and adapters must match.
Q3O is published only after all 192 selected-song scores exist; partial scores
remain diagnostics. Sharded Q3O resets request sequence between chunks and is
explicitly a local-checkpoint extension, not exact full-sequence paper acceptance.
Completed Gadi chunks are packed into private uncompressed tar archives with
per-member and whole-archive SHA256 verification before any materialized copies
are removed. Generation metadata and score chunks remain available for merging;
incomplete chunks are never cleaned up. This avoids exhausting the projects'
scratch inode quotas without deleting the original audio or raw scoring receipts.
The evaluator venvs/runtime and large asset/cache files are spread across the
two mounted scratch projects; existing research folders are not modified. The
independent SongBench eval environment resides on iv96, while the metrics venv
and portable Python runtime remain on wa66; their logical venv paths are unchanged.
After a shared scratch inode-limit failure, the preparation job moves only its
own UV download/extraction cache to 150GB allocation-local jobfs. Installed
environments use the original `UV_LINK_MODE=copy`, so removing that temporary
link after preparation does not remove runtime packages. Failed dependency
jobs are resubmitted, not repaired only through `qalter`: PBS retains the old
upstream failure relationship. No original frozen evaluator code/locks change.
Gadi's host glibc 2.28 is also older than the frozen [vLLM 0.14.0 wheel's
glibc 2.31 minimum](https://pypi.org/project/vllm/0.14.0/#files). The campaign
uses a hash-bound Python 3.12.11 Debian bookworm OS image; native /apps Python
and its existing Torch/YuE2 packages are preserved through explicit compatibility
library bindings. Frozen scorer locks and model hashes remain unchanged.
`container-runtime.json` stores the SIF identity and actual import preflight.
The image includes GCC and a compiler preflight: the earlier slim image could
import the native runtime but failed to build the frozen madmom extension.

No cron, scheduled task or automation is created. The user requests subsequent
recovery manually. Submission receipts are not a live scheduler-status claim;
shared queue wait times are not predictable enough for a guaranteed ETA.
