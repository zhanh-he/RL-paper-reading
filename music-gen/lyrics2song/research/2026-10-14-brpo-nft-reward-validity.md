# BRPO, NFT and the gap between reward ranking and policy improvement

Primary-source discussion updated 2026-10-14. No new experiment, model inference,
reward scoring or training is launched by this reading pass. Existing approved
work and frozen data protocols are unchanged. This is not an algorithm leaderboard.

## Decision

Accurate **offline ranking of a specified attribute** does not establish accurate
ranking of overall utility on future policy outputs. Distinguish incomplete
objectives, all-bad groups, tail errors, policy shift, candidate support and update
implementation. A rise in one scorer with another declining is a proxy conflict,
not automatically human-confirmed reward hacking.

DPO offers an auditable preference-data interface, not immunity to noisy labels.
BRPO changes comparison/advantage construction; NFT changes the learning operator.
Neither creates a correct perceptual objective.

## A local implementation qualification: two candidates

At repository baseline
[7ab16b263cca0f2bafbd92dcd1ea874b11fb8700](https://github.com/zhanh-he/RL-paper-reading/tree/7ab16b263cca0f2bafbd92dcd1ea874b11fb8700),
the [YuE2 regime trainer](https://github.com/zhanh-he/RL-paper-reading/blob/7ab16b263cca0f2bafbd92dcd1ea874b11fb8700/music-gen/reward-regime-study/train_yue2.py#L129)
uses two candidates and population-standard-deviation normalization.

For a nonzero score gap outside the numerical floor:

$$
A_1=\operatorname{sign}(r_1-r_2),\qquad A_2=-A_1.
$$

This is our algebraic reading, not a new numerical experiment. Tiny uncertain
wins and large clear wins receive similar weight. Replacing the same scalar
ordering by a win/loss label may add little information. General larger-group
GRPO does retain magnitude information; do not extend the two-candidate result
to every trainer or framework.

The current regime code matches codec/EOS sampling support; older pilot
full-vocabulary scoring is a separate version. Fresh single-step old/current
ratios are expected to start at one if forwards match, so clipping is not a hard
bound on the resulting optimizer step. This is a source-based expectation, not a
measured clip fraction or a diagnosis of every run.

## Writing-Zero and BRPO

The paper combines a rubric/critique-based pairwise GenRM with a randomly chosen
current-group reference. Eq.11 gives +1 only when all voting conditions favor the
candidate; otherwise -1. Ambiguity is not automatically a neutral label. Filtering
mixed/one-sided groups does not establish judgment correctness.

The higher-accuracy offline scalar RM performs worse in the tested downstream
scalar-GRPO system than GenRM+BRPO. Judge and optimizer both change, however:
this does **not** isolate the causal benefit of BRPO. Human comparisons are for
Writing-R1 and include many ties.

A dynamic comparison reference is not an updated reward model. Zero-expectation
intuition requires explicit symmetry, tie, self-comparison and aggregation
conditions; rejecting an unconditional guarantee is not evidence of an
implementation bug. A nonzero average advantage alone also does not prove a
biased policy gradient. [Writing-Zero v2, §3/§5, Tables 1–3](https://arxiv.org/html/2506.00103v2)

Borrow the comparison protocol first: same prompt, shuffled order, clear rubric,
abstention/ties, absolute winner acceptability and non-target quality. A full BRPO
comparison should hold the judge fixed. A custom sign-GRPO is not a reproduced BRPO.

## DiffusionNFT and AR-NFT are different interfaces

DiffusionNFT scores final samples with a black-box reward, forward-noises clean
latents and fits implicit positive/negative velocity policies. It avoids reverse
trajectory likelihood ratios. The continuous denoiser, latent target and noise
schedule must be available; it is not a drop-in loss for an AR codec-token actor.
The reward need not be differentiable, but wrong high scores still drive wrong
learning. Reward weights in [0,1] are not calibrated human probabilities.
Sampling-policy EMA does not refresh the evaluator. Image speed results are not
audio measurements. [DiffusionNFT v2, Algorithm 1](https://arxiv.org/html/2509.16117v2)

Read-only source inspection at official commit
[cbb14f8](https://github.com/NVlabs/DiffusionNFT/tree/cbb14f84b8312b620390dfcbe2fab69c1383b104)
also distinguishes interpolation beta from the training anchor coefficient,
adaptive x0 weighting from schematic velocity loss, and velocity-MSE anchoring
from exact probability KL. Sampling-old-policy EMA and evaluation EMA are
different objects.

The separate AR paper, *NFT: Bridging Supervised Learning and Reinforcement
Learning in Math Reasoning*, proves an on-policy gradient equivalence to GRPO
under binary rewards, matching prompt weighting and ratio-one assumptions.
It does not prove equivalence for arbitrary music feedback or multiple update
epochs. A different loss name need not fix the first-step direction.
[NFT v3, Proposition 4.2](https://arxiv.org/html/2505.18116v3)

## Route priority, not authorization

| Route | Priority | Potential value | Boundary |
|---|---|---|---|
| Current-policy reward qualification | P0 | Separate ranking, absolute quality, tail and shift validity | Needs independent utility, not only another correlated scorer |
| Curated/non-inferior DPO and selective GRPO | P1 | Auditable feedback and abstention; matched-feedback control | Wrong labels and absent good candidates remain |
| Semi-online candidate refresh | P1 | Reduce stale preference support | Fixed evaluator may still fail on new outputs |
| BRPO comparison protocol | P1 | Conditional comparisons, order checks and rubric | Votes are not calibrated confidence |
| Full pairwise GenRM + BRPO | P2 | Alternative feedback/advantage system | Hold judge fixed to attribute optimizer effects |
| AR-NFT, GIFT, ADPO | P2 | Negative-aware or implicit/listwise alternatives | No demonstrated cure for music reward semantics |
| DiffusionNFT vs Flow-GRPO | Conditional P1/P2 | Continuous-actor optimization alternatives | Requires real trainable flow interface and matched budget |
| Evaluator/teacher refresh | P2 | Address changing evaluator-state distribution | Not replaced by sampling EMA; cost and independent gold remain |

Confidence should estimate independent pair-direction correctness conditional on
the task/policy, and absolute winner acceptability separately. Detector agreement,
reward spread, judge repeatability and teacher entropy are diagnostics, not that
probability.

## Additional adjudications

- GIFT v5 uses stop-gradient normalization statistics and includes the
  sampling-distribution score-function term. It is not simply ordinary MSE
  replacing policy gradients. GIFT and ADPO first appeared in 2025; stable catalog
  IDs retain their historical names. ADPO's v7 PDF and abstract title variants
  identify one paper. [GIFT](https://arxiv.org/html/2510.23868v5),
  [ADPO](https://arxiv.org/html/2510.18913v7).
- Robust DPO is ICML 2024. Its random-label-flip assumptions do not cover arbitrary
  systematic MIR errors. [Formal paper](https://proceedings.mlr.press/v235/ray-chowdhury24a.html).
- MR-FlowDPO's paper describes music-HuBERT semantic scoring; the released
  [implementation](https://github.com/lonzi/mrflow_dpo/blob/1086a32b226f330779f6604981c3337e694dbbcc/mrsd_dataset/semantic_consistency_reward.py)
  uses MusicFM and different aggregation. Keep paper and released recipe separate.
  [Paper §3.3.3–3.4](https://arxiv.org/html/2512.10264v2).
- FlowSE explicitly reports single-objective degradation in other automatic
  metrics. Its composite is normalized and unequally weighted; it does not prove
  all trade-offs are solved. [§4.1.1–4.1.2](https://arxiv.org/html/2601.16483v1).
- The perceptual speech-enhancement paper uses GSPO, not GRPO. Its equal-weight
  composite has human support, not a general non-regression guarantee. Exact
  single-metric preference numbers differ between narrative and Table 3;
  prioritize the table and disclose that discrepancy.
  [Speech paper](https://arxiv.org/html/2606.21458v1).
- LeVo 2, Qwen-Music and Full-Song support staged routes, not universal optimizer
  ordering. Candidate budgets, score scales, renderer branches and unisolated
  OPD effects stay distinct. [Detailed reading](2026-10-14-post-training-reading.md).

## Citation jobs

[Rethinking Reward Model Evaluation](https://proceedings.iclr.cc/paper_files/paper/2025/hash/93f250215e4889119807b6fac3a57aec-Abstract-Conference.html)
supports downstream-oriented RM evaluation, with controlled model/synthetic gold,
not musical human ground truth.
[GAPT](https://proceedings.iclr.cc/paper_files/paper/2026/hash/e6731a1d3eddafeb8c94669a1d82fc5d-Abstract-Conference.html)
provides a music-specific coherence/diversity failure setting, not full-song vocal
alignment.
[DeepSeekMath](https://arxiv.org/html/2402.03300v3) is the GRPO method source.

The strongest research framing is **optimization-aware reward qualification**,
followed by independently verified selective/non-inferior feedback. Do not
precommit to BRPO/NFT as the answer before matched evidence.
