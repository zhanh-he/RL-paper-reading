# SongBench Mean7: Priority Experiment Plan

2026-10-20 AWST. **Execution approved. First complete Muse's missing original-base SP/subhead controls; then run the Mean7 priority matrix.**

[Online plan and recovered results](https://zhanh-he.github.io/RL-paper-reading/demos/#lyrics-model-plan) | [Audited historical receipts](runs/2026-10-20-songbench-review/verified-recovery.json)

## The Question and the Exact Objective

Can optimizing one SongBench scalar improve generated songs, and does the answer differ between YuE2 and DiffRhythm2? Instrument-count adherence, overall audible quality, clipping and lyric fidelity remain independent observations, not implied benefits of a larger optimized score.

The user selected **seven-dimensional arithmetic mean as the only reward**:

`R = (Vocal + Instrumental + Melody + Structure + Arrangement + Mixing + Musicality) / 7`

These are the seven raw heads from the frozen SongBench scorer. `Instrumental` is the code name of the published Instrument axis. This arithmetic scalarization is **our declared objective**, not a claim that the SongBench authors prescribe a canonical RL aggregate. No per-head reweighting, clipping, hidden normalization, SP term, SongEval term or loudness/clipping penalty is added. Keep the native score scale and log all seven raw scores and their exact mean.

Primary-source check: [official seven-axis definitions](https://github.com/Tencent/SongBench/blob/9361b55ebff3fffcefbbc9a4653936ae6789c3c8/README.md) and [official head mapping / full-file scoring](https://github.com/Tencent/SongBench/blob/9361b55ebff3fffcefbbc9a4653936ae6789c3c8/eval.py), checked2026-10-20. The historical scorer is our archived unrounded adapter around the frozen head, not a claim that its wrapper is byte-identical to the current official CLI; new work must pin both the MuQ/frontend and head checkpoint identities.

`loss = GRPO_policy_loss + beta * reference_KL`, initially **beta=0.05**. KL is a separate loss term, not an eighth reward. For G2 with standardized advantages, multiplying the entire reward by a positive constant does not usually alter ranking/advantages; it does not make differently defined KL units comparable. Record zero-variance groups and skip only their policy-gradient contribution, not their existence.

A failed decoder, missing score, nonfinite audio or scorer exception is a technical failure: retain the receipt and retry a bounded number of times with the same seed, then stop the arm. Do not silently assign it zero, replace a difficult condition, or omit it. Finite quiet/clipped audio is still scored and reported; there is no hidden safety penalty in this pure-reward experiment.

## What We Have, and What We Do Not

The read-only recovery checked all six YuE2/Diff historical arms: contiguous270 updates, final adapter/optimizer/RNG checkpoint files and metadata, first-update reference identity, complete30-condition validation at0/50/100/270, same-model shared baseline, and all3,960 train/validation audio SHA bindings. The projected data hash is `c39d43e81793bee3f312c7028a0483b5da41068db4544acd258115043101e46f`. The SongBench head hash is `ca27966a6d2b3312587b9e1d53ee3adf6bde8ac68a7d339e6f962206cbfde312`. Remote core/reward code hashes match the archived source. This is filesystem/log validation, not a fresh GPU checkpoint reload or fresh independent listening study.

**No verified historical run trained this Mean7 objective.** All six optimized either SP Richness, Instrumental or Arrangement, with a disclosed clipping/loudness penalty. Their seven-head mean below is a retrospective cross-score on the same saved validation audio, not the reward they optimized.

| Historical model / objective | Train | Valid at0/50/100/270 | Mean7 at0 | Mean7 at270 | Change |
| --- | --- | --- | ---: | ---: | ---: |
| YuE2 / SP + guard |270|30/30 each|5.391|5.165|-0.226|
| YuE2 / Instrument + guard |270|30/30 each|5.391|6.003|+0.612|
| YuE2 / Arrangement + guard |270|30/30 each|5.391|5.747|+0.356|
| Diff / SP + guard |270|30/30 each|5.116|5.077|-0.039|
| Diff / Instrument + guard |270|30/30 each|5.116|5.161|+0.045|
| Diff / Arrangement + guard |270|30/30 each|5.116|5.228|+0.112|

Interpretation: enough signal to justify testing Mean7, not proof of audible improvement or a universal high/low-MOS regime. These are30-condition validation descriptions from one training seed. YuE2 used rank4 q/v LoRA, beta0.01 and short200-600 semantic-token rollouts; Diff used rank8 attention+MLP, beta0.05 and native60s budgets. Diff perturbed one selected block with two stochastic transitions, rather than treating the full song as one discrete AR trajectory. Cross-model effect sizes therefore do not isolate model quality. Head-only failed runs, mock runs, incompatible data protocols and partial validation means are excluded from the primary recovery.

Muse and HeartMuLa's six new backbone submissions failed before training: scorer startup could not import `generate_objective` through `enter_eval_container`. They are **not still queued and not completed**. Preserve the failure metadata, not spurious training scores. SongBloom has only old failed head runs/isolated acceptance; ACE has no verified formal RL result here.

## Priority and Scope

### Approved Muse Completion, First

The user explicitly restored the missing Muse comparison as **P0**, before expanding the Mean7 matrix. Run three fresh arms from the original **bolshyC/Muse-0.6b**, revision `0fb6b5b0a33ec48d135f046ea09524e52e502a42`, weight SHA256 `c810b2a748579e06e960019f89a9cb98701e4ff545e2c760e8ceaa3343ff1b52`. The same bytes were checked on5090 and Gadi. Never use a prior GRPO checkpoint, merged GRPO weights or old adapter as the baseline/start. Each launch checks this identity and starts a new backbone LoRA (rank8/alpha16,5,046,272 trainables).

The three targets are SP Richness, SongBench-Instrumental and SongBench-Arrangement, **each minus the disclosed old guard**, matching the missing family of controls. These are not the new pure Mean7 target. Fix LR3e-4, constant schedule, beta0.05, G2, seed42 and270 updates, all270 train and30 validation conditions. Retain the old native512-completion-token/10.24s fixed-seed MuCodec decode protocol; do not quietly call these60s-trained or a controlled equal-budget cross-model comparison. Baseline/final validation uses all30 conditions; no partial mean is a final result. Long playback is a distinct inference task after checkpoint recovery, not fabricated by looping the10.24s training clips.

The repaired Muse path passed two real backbone updates on5090: first KL0, second KL0.000505387, nonzero gradients on both steps, all four scored PCM24-FLAC SHA bindings checked, and adapter/optimizer/RNG files saved and hashed. This is an acceptance test, **not** a formal result or a cold-restore proof. The original server-mode setup had same-GPU NCCL failure; native colocated vLLM, an explicit1GiB KV cache and codec/actor allocator-cache release resolved the tested failures without changing decoder math. All300 data rows are verified single-user-turn messages, so no server-only multi-turn scheduler is required.

Current one-shot launch evidence is in [launch.json](runs/2026-10-20-songbench-priority/launch.json). Muse SP runs on5090, Instrument queues on Gadi wa66/Hopper, Arrangement on iv96/A100. Across hosts, confirm/recover a canonical baseline before publishing a common A/B audio baseline; a common checkpoint/seed alone is not proof of byte-identical audio. HeartMuLa, SongBloom and ACE remain deferred.

| Priority | Experiment / action | Why now | Acceptance / decision |
| --- | --- | --- | --- |
| P0 execution | Complete Muse original-base SP / Instrument / Arrangement controls | Explicit user priority; missing comparison model | New original-base identity checks,270 updates, full30 baseline/final, Reward/KL; no previous GRPO start |
| P0 recovery, done | Retain the six verified historical arms; publish30/30 means, traces and fixed-index existing MP3s | Gives a trustworthy baseline and diagnostic comparison without GPU work | Clearly historical subhead/SP + guard, never relabeled Mean7 |
| P1 first | **DiffRhythm2–SongBenchMean7–3e-4**; Miles native backbone LoRA | Accepted real training lifecycle, native60s, and newly recovered endpoint; smallest integration gap | Same accepted stochastic-window policy, rank8/alpha16, G2, beta0.05,270 updates, seed42; Mean7 scorer adapter is new and must pass identity/logprob/gradient/reload checks |
| P1 paired anchor | **YuE2–SongBenchMean7–3e-4**; TRL native-policy adapter | Stronger-start model plus substantial historical learned-head response | Rank8/alpha16 attention+MLP, G2, beta0.05,270 updates, seed42. Existing YuE2 is custom PyTorch, NOT accepted TRL: first validate the native codec/EOS masks, old/new policy support, fixed reference and actual backbone gradients; no claim this port already works |
| P2 | **Muse–SongBenchMean7–3e-4** after the three original-base controls | Small AR model could test whether the result extends beyond YuE2 | Pure Mean7 needs its own fail-closed reward adapter, not the guard-inheriting old Muse class; all270/30 inputs, native codec seed, likelihood/length audit. Not submitted yet |
| P3 | HeartMuLa Mean7; then SongBloom/ACE only if capacity remains | More codec/flow integration and runtime risk; not necessary for the first main claim | Full native codebooks/backbone accepted first; do not resurrect head-only surrogates or equate isolated smoke with a completed experiment |

Do not start a broad LR grid, all-seven single-head training grid, or0.6/4B/XL speed sweep now. The decision is about one Mean7 objective. Existing Instrument/Arrangement are historical ablations; an eventual matched one-head comparison must use the same new model, adapter, beta, duration, seed and training budget. A raw-Mean7 versus guarded-Mean7 ablation is a later separate experiment, never a hidden alteration of the primary objective.

## Data and Matched Evaluation

- Keep the existing downloadable frozen manifest: **270 CMI train /30 valid /121 CMI test**, plus **192 WildSongBench** external conditions. Do not go back to469 source rows or resplit. Those rows include duplicates/overlap and are not the frozen unique training units. Check the model-specific projection SHA and membership, not just counts. Retain the existing9:1 train/valid rule and decontamination provenance.
- Models consume native inputs. YuE2 uses text+lyrics; Diff style text+lyrics; reference audio remains available in the source manifest for future native-audio models, not secretly added to these policies.
- Target training/validation60s native output budgets for the paired primary runs. YuE2's60s path is a **new memory/likelihood acceptance requirement**, not established by its old600-token training. Use native EOS with an explicit maximum budget and record actual lengths; early ends stay early and are reported. Never pad, loop or trim until a score improves. If native60s training fails memory acceptance, propose the exact shorter protocol for approval rather than silently changing it.
- Full validation:30/30 at0/50/100/270 with fixed seed5101+index, common frozen per-model baseline and all seven dimensions, Mean7, SongEval, lyric error where applicable, peak/RMS/clipping/duration. Save intermediate1/5/25 checkpoints for diagnostics/replay if practical; they do not replace full checkpoints.270 is the preregistered primary endpoint. Any valid-selected checkpoint is secondary and selection uses validation only.
- First complete the full121-condition CMI test on frozen baseline/final checkpoints, one pre-fixed candidate seed per condition, paired by input. Then192-condition WSB external evaluation after the configuration/analysis rule is frozen. Keep any official multi-candidate/PER-selected WSB protocol as a separately labeled secondary protocol; do not mix best-of-two with the unselected primary. Metrics not supported for all languages/modalities are n/a with a reason, never partial-language means labeled full-cohort results. These datasets have been used in earlier campaigns; do not claim they are newly discovered, pristine blind test sets.
- Mean7 is an optimized metric, not an independent judge. Primary reporting must include independent SongEval and paired blinded listening; retain the seven-head breakdown to see if the mean hides a damaged dimension. Predicted scores do not establish MOS or instrument count. Run a paired human study on30 validation conditions for the early readout, randomized order and model identities; report response N and uncertainty. Minimum two ratings/pair is a pilot, not a formal high-precision MOS claim.

## Demo and Failure Analysis

Five examples per accepted new experiment: three frozen native validation cases and two controlled prompts with explicit low/high instrumentation counts, compared with the same lyrics/seed without a count constraint. These modified probes are excluded from formal validation/test means. Display full inputs where redistributable, one frozen baseline,0/50/100/270 outputs, matching mel/wave scales, MP3 playback, Reward on the left and correctly labeled KL on the right. Native60s is the first target;300s is optional later, not a reason to delay the primary result.

Predeclare diagnostic selection: target score up but independent score/listener preference down; one SongBench head down while mean up; worse clipping/lyric error; duration collapse; or ignoring a requested sparse arrangement. Retain all-condition tables. A conflict is a **candidate badcase**, not proven reward hacking until listening/independent evidence supports it. Instrument predicts realism, Arrangement orchestration, neither is a literal instrument counter.

## Resource and Approval Boundary

Use the5090 first for Muse SP. A finite dependency queue starts Diff Mean7 only after Muse's training and full validation finish successfully; it does not schedule recurring work or interrupt another job. Gadi wa66/iv96 hold the two other Muse controls and a YuE2 Mean7 gated job. Priority1023 is the submitted scheduler priority, not a guarantee of immediate allocation. No duplicate submissions for an already owned arm. Queued requests are not results and no ETA is promised without actual allocation/rate evidence. Do not disturb unrelated CASM/LaDA work. No scheduler automation or monitoring task.

**Approved execution:** Muse original-base three-arm controls first; Diff and YuE2 Mean7 next,3e-4/beta0.05/G2/rank8 for270 updates. Diff repeats its real Miles acceptance and cold-actor optimizer/RNG restore with the new Mean7 scorer. YuE2's native TRL imports/tokenizer pass on5090 and Gadi, but GPU/60s acceptance remains outstanding: two real native updates, a separate process continuing the saved step2 checkpoint to3, then a fresh270-step formal run. Failed gates stop the job; never replace them with a head-only or shorter surrogate. Resume-to3 is a continuation gate, not yet a proven bit-exact uninterrupted/resumed equivalence test. New Mean7 has no completed formal result at submission time.
