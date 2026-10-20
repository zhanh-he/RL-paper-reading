# SongBench Mean7: Priority Experiment Plan

2026-10-20 AWST. **Reward definition confirmed by the user; new training awaits approval.**

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

| Priority | Experiment / action | Why now | Acceptance / decision |
| --- | --- | --- | --- |
| P0 recovery, done | Retain the six verified historical arms; publish30/30 means, traces and fixed-index existing MP3s | Gives a trustworthy baseline and diagnostic comparison without GPU work | Clearly historical subhead/SP + guard, never relabeled Mean7 |
| P1 first | **DiffRhythm2–SongBenchMean7–3e-4**; Miles native backbone LoRA | Accepted real training lifecycle, native60s, and newly recovered endpoint; smallest integration gap | Same accepted stochastic-window policy, rank8/alpha16, G2, beta0.05,270 updates, seed42; Mean7 scorer adapter is new and must pass identity/logprob/gradient/reload checks |
| P1 paired anchor | **YuE2–SongBenchMean7–3e-4**; TRL native-policy adapter | Stronger-start model plus substantial historical learned-head response | Rank8/alpha16 attention+MLP, G2, beta0.05,270 updates, seed42. Existing YuE2 is custom PyTorch, NOT accepted TRL: first validate the native codec/EOS masks, old/new policy support, fixed reference and actual backbone gradients; no claim this port already works |
| P2 | **Muse–SongBenchMean7–3e-4** after packaging repair | Small AR model could test whether the result extends beyond YuE2 | Repair scorer imports without changing shared env; all270/30 inputs, fixed native codec seed, same likelihood/length audit. No need to resume superseded subhead jobs |
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

Use the5090 for the first accepted Diff arm if idle; YuE2 and Muse can use independent Gadi wa66/iv96 or Kaya allocations after their exact path is accepted. Only short debug sessions for acceptance, followed by genuine full270-step jobs. Queued requests are not results and no ETA is promised without actual allocation/rate evidence. Do not disturb unrelated CASM/LaDA work. No scheduler automation, no monitoring task, no new submission from this planning update.

**Approval requested:** run the two P1 Mean7 arms, Diff first and YuE2 after native-TRL/60s acceptance, at3e-4/beta0.05/G2/rank8 for270 updates; repair Muse packaging and then make it the next model. The current approval covers the reward definition only, not this new execution matrix.
