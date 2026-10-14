# Lyrics-to-song post-training: source-checked reading map

Checked on 2026-10-14 against the papers' methods, relevant results and
appendices, plus official release documentation. These are author-reported
results, not reproduced measurements. Personal catalog ratings are unchanged.

## Published Recipes, Not A Universal Pipeline

| Work | Publicly described route | Important boundary | Primary source |
| --- | --- | --- | --- |
| LeVo | Pretraining, modular extension, dimension-specific DPO, parameter interpolation | Separate lyric, prompt and musicality preferences; not one joint scalar objective | [Paper, section 3.4](https://arxiv.org/html/2506.07520v3) |
| LeVo 2 | SFT, offline controllability DPO, periodically refreshed semi-online musicality DPO, modular extension | Semi-online DPO is still pairwise preference learning, not GRPO | [Paper, III-E and Tables II-III](https://arxiv.org/html/2606.30642v1) |
| HeartMuLa | SFT, dimension-specific DPO, model merging | Chinese SongEval Avg falls 4.46 to 4.40 while PER improves 0.1067 to 0.0806; English Avg improves 4.41 to 4.43 | [Paper, section 3.4.4, Tables 14-16](https://arxiv.org/html/2601.10547) |
| DiffRhythm+ | Quality SFT and diffusion-DPO | SongEval for songs; Audiobox for instrumentals with a scale mapping. Numerical rescaling alone does not establish equivalent preference calibration | [Paper, III-D](https://arxiv.org/html/2507.12890v2) |
| DiffRhythm 2 | Block-flow training, cross-pair preference optimization and model merging | CPPO couples musicality/lyrics and style/quality; inference release is not a complete training recipe | [Paper, section 3.4](https://arxiv.org/html/2510.22950) |
| ACE-Step 1.5 | Generator LM GRPO, DiT intrinsic alignment and distillation; captioner RL is separate | Main Table 1 explicitly uses a no-RL configuration, so its scores cannot isolate RL gains | [Paper, sections 2 and 4, Table 1](https://arxiv.org/html/2602.00744) |
| Qwen-Music | Supervised cold start, iterative offline DPO, online GSPO | GSPO is not GRPO; the route does not establish that every model must use DPO first | [Paper, section 3.2](https://arxiv.org/html/2607.11699) |
| YuE2 | AR pretraining, AR-NAR joint training, annealing | No generator DPO/GRPO stage is reported. Two-candidate and best-of-8 selection are inference budgets, not weight updates | [Paper, sections 2 and 4, Appendix A.3](https://arxiv.org/html/2609.33757v1) |
| Alibaba Full-Song | DPO + chosen SFT, GRPO + online RSFT, then GRPO + OPD; separate FullDiT flow-based GRPO | No matched LM ablation isolates OPD's incremental music-quality gain | [Paper v3, section 2.6](https://arxiv.org/pdf/2607.20253v3) |

Commercial products without a disclosed recipe remain **undisclosed**, not
evidence of training with or without RL. Different models, data, codecs and
candidate budgets do not constitute an optimizer comparison.

## Three Checks That Change The Interpretation

1. **LeVo 2's OVL is expert MOS on a 1-10 scale**, not 1-5. Offline to
   semi-online DPO changes OVL from 5.12 to 5.37 and PER from 9.19% to 9.22%.
   Its README advises against supplying both description tags and reference
   audio, especially contradictory conditions; the paper nevertheless includes
   joint-condition DPO samples. This is not an architectural impossibility.
   [Evaluation](https://arxiv.org/html/2606.30642v1#S4),
   [official input guidance](https://github.com/levo-demo/LeVo).
2. **YuE2 selection is consequential.** Appendix A.3 first removes candidates
   jointly dominated in Musicality, Q3O and PER, applies two 0.10 score windows,
   then selects lowest PER. The Table 1 PER changes from 0.0844 to 0.0979 under
   best-of-8 even as SongBench Global Avg rises. This is not uniform improvement
   or a clean post-training effect. [Protocol and results](https://arxiv.org/html/2609.33757v1).
3. **Alibaba versions must be pinned.** Downloaded v2/v3 PDF text differs in
   the version stamp and the Introduction's external evaluation count:
   2,105 in v2, 2,289 in v3. The checked method equations and result tables are
   unchanged. These are leaderboard snapshots, not current live counts.
   [v2 PDF](https://arxiv.org/pdf/2607.20253v2),
   [v3 PDF](https://arxiv.org/pdf/2607.20253v3).

## OPD: Dense Supervision Is Not A Verified Reward

In Full-Song, a frozen SFT teacher supplies full-vocabulary distributions on
student-generated prefixes. Equation 17 uses teacher-to-student forward KL;
Equation 18 adds it to GRPO, without an explicit RSFT term. On-policy describes
the prefix distribution, not a mandatory KL direction. Teacher, rollout-old
policy and fixed reference policy are distinct roles.

- [On-policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://openreview.net/pdf?id=3zKtaqxLhW),
  ICLR 2024, introduces GKD, allows different divergences and combines
  distillation with RL. It is a language-model precedent, not a music experiment.
- [On the Off-Policy Teacher in On-Policy Distillation](https://arxiv.org/html/2609.38360v1),
  2026 preprint, studies unreliable teacher continuation on student prefixes.
  SCOUT periodically adapts the teacher using verifiable outcome rewards.
  Its reasoning/code results do not establish a valid verifier for music.

FullDiT renderer ablations and the full system's final rankings cannot isolate
OPD. Reusing SongBench for training and evaluation calls for independent
assessment, but does not by itself prove leakage or reward hacking.

## Metadata Corrections

- ACE-Step 1.5 section 4.1 reports 50 to **8** sampling steps at Shift=3.
  Its A100 timing is kept as an author-reported renderer claim, not end-to-end
  planner latency. [Source](https://arxiv.org/html/2602.00744).
- WildSongBench is a benchmark resource; the generator's 3.58B parameter count
  belongs to YuE2, not the dataset.
- DiffRhythm 2's paper points to the [Xiaomi research repository](https://github.com/xiaomi-research/diffrhythm2).
  The [ASLP-lab release fork](https://github.com/ASLP-lab/DiffRhythm2) remains
  available and links its weights; retain the relationship instead of calling
  it a broken or unrelated repository.

No new experiments, human scores or optimizer superiority claims are introduced
by this reading update. The [interactive literature catalog](https://zhanh-he.github.io/RL-paper-reading/)
is the shared metadata entry; the [research map](README.md) describes native
interfaces and reproduction boundaries.
