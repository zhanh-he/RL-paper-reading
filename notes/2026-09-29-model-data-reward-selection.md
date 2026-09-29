# Model, data and reward decisions for the three demos

Evidence state: checked 2026-09-29. `Run` means executed locally; `staged` means code or models present without a measured training result. A downloadable checkpoint is not permission to use it in a commercial setting.

| Case | Model | Public weights and fit | Local status | Decision |
| --- | --- | --- | --- | --- |
| SATB / multi-instrument | [MuScriptor](https://github.com/muscriptor/muscriptor), 103M/307M/1.4B | Published weights are [gated on Hugging Face](https://huggingface.co/MuScriptor) and noncommercial; strong multi-instrument event output | Code/environment staged; weight request returns 403 | Best event-level GRPO target after access and license approval. Do not claim a local MuScriptor result. |
| SATB | Local ChoralStream | Existing user-owned checkpoint and YouChorale-Pro splits | **Run:** frozen-encoder Bernoulli frame-head GRPO, 96 train / 28 valid / 30 test songs | Presentation's local causal pilot, but it does not optimize MIDI notes. |
| Multi-instrument | [Slakh2100](https://www.slakh.com/) | 2,100 aligned audio/MIDI multitrack songs, CC BY 4.0; synthesized and known to contain some replicated MIDI | Not downloaded for this pilot | Pick 64/128/300 distinct source songs by MIDI identity; reserve test at song level and avoid duplicate MIDI. Use held-out real recordings for transfer, not only synthetic Slakh. |
| Four-part transcription | [CocoChorales](https://magenta.tensorflow.org/datasets/cocochorales) | SATB-style four-part instrument ensembles with aligned MIDI, CC BY 4.0; **instrumental, not choral singing** | Not downloaded | Reward unit tests and transfer stress test, not a substitute for vocal-choir test data. Tiny subset is adequate for smoke. |
| Vocal-conditioned generation | [ACE-Step 1.5 base](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/INFERENCE.md) | Official `complete` task for base model; [weights MIT](https://huggingface.co/ACE-Step/acestep-v15-base) | **Run:** 16 s synthetic guide to 16 s completion, seed 29 | Playable feasibility demo only. Output may be full mix and there is no GRPO result. |
| Vocal-only accompaniment | [AnyAccomp](https://github.com/AmphionTeam/AnyAccomp) | Dedicated accompaniment, [weights CC BY 4.0](https://huggingface.co/amphion/anyaccomp), code MIT | Repo cloned on lab5090; no inference yet | Better task contract than full-mix completion. Upstream pins Torch 2.3.1, incompatible with RTX 5090; adapt to newer Torch before demo claims. |
| Vocal-conditioned generation | [LaDA-Band](https://github.com/Duoluoluos/LaDA-Band) | Vocal-to-accompaniment code; verify checkpoint access and usage terms | Not deployed | Candidate only, not tomorrow's measured result. |
| Text/lyrics to song | [Muse + MuseCritic](https://github.com/WuqnEl/MuseCritic) | Official reward/GRPO recipe and pretrained Muse artifacts already staged on Gadi | One-step GRPO smoke queued on H200 | First text-to-song online reward experiment if job runs. One step is engineering proof, not a quality delta. |
| Text/lyrics to song | [YuE2](https://github.com/multimodal-art-projection/YuE) / [Tencent LeVo2](https://github.com/levo-demo/LeVo) | Stronger open-weight song generators; respective training and commercial terms require review | YuE2 inference preparation exists; LeVo2 not deployed | Quality/model-size comparison later. Neither is a measured GRPO arm here. |

## Data budget for a defensible follow-up

Use the **same** source-disjoint train set of 64, 128 and 300 examples for each reward arm, plus a fixed validation set and untouched held-out test. Our immediate ChoralStream pilot used 96 unaugmented training songs, one 5.12-second clip each, and 30 held-out songs. The `30-song` paired bootstrap quantifies variation across those songs, not variation across training seeds or full-song MIDI accuracy.

For a real vocal-to-accompaniment comparison, gather a small authorized set of clean, fixed vocals with separated accompaniment reference and beat annotations; hold out singer/song identities. Do not score an ACE-Step full-mix completion as an isolated accompaniment. For Muse, start from the [official MuseCritic prompts](https://github.com/WuqnEl/MuseCritic) and keep prompt/artist/seed splits fixed; never train on benchmark prompts.

## Reward experiments

| Case | Single-arm GRPO rewards | Independent endpoints and failure guards |
| --- | --- | --- |
| MIDI transcription | Track-aware pitch+onset F1; track-aware pitch+onset+offset F1; per-voice recall floor; fragmentation penalty as auxiliary | Held-out note/track F1, per-voice precision and recall, onset/offset error, duplicate/octave overfill. No unconditional “four voices active” or raw note-count reward. [Four-pair controlled audit](../music-trans/choral-singing/rewards/audits/2026-09-29/README.md) demonstrates why. |
| Fixed vocal -> accompaniment | Beat v2/v5; coverage; richness; then a guarded combination after scale and preference checks | Fixed-vocal similarity, alignment, accompaniment isolation, blind A/B, reward hacking cases. No measured GRPO yet. |
| Lyrics/style -> song | SongEval; MuseCritic mean/five heads; CMI-RM (not CMI-RewardBench itself); music aesthetics, each isolated first | Frozen independent SongEval/Audiobox, lyrics/style adherence and blind human A/B. A reward cannot validate itself. |

For size/quality effects, stratify each pretrain checkpoint by `support@8` (any usable candidate among eight) and reward pairwise validity **before** GRPO. A tiny model with no usable samples may get no gain; a strong model may improve proxy scores but regress on held-out quality; a weak but sample-capable model is the best chance for visible improvement. These are hypotheses, not results from our one-checkpoint pilot.
