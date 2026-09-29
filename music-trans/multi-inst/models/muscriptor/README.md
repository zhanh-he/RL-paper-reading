# MuScriptor

Upstream: [paper](https://arxiv.org/abs/2607.08168), [code](https://github.com/muscriptor/muscriptor), [model organization](https://huggingface.co/MuScriptor). The released checkpoints are gated and noncommercial; this directory is an adapter slot, not a mirror of weights or a claim that training is available. Record the exact checkpoint and approval before adding a run. The public repository's final checkpoints do not provide a clean pre-RL baseline for reproducing its paper's pre/post comparison.

On 30 Sep 2026, we submitted our original 10.2-second synthetic SATB example to the [official hosted demo](https://muscriptor.kyutai.org/) and downloaded its MIDI and SoundFont-rendered audio. The [receipt](muscriptor_receipt.json) and [demo](https://zhanh-he.github.io/RL-paper-reading/demos/#choral) show 16 notes on one acoustic-piano track, with pitch-only 50ms onset F1 0.667 and onset+offset F1 0.500 for this single example. The online model did not provide SATB labels, so there is no track-aware score. This is an inference baseline, not a GRPO result or held-out benchmark.

The lab5090 Hugging Face account currently receives `Access denied. This repository requires approval` for `MuScriptor/muscriptor-small`; local weight-based training remains blocked until that account accepts the model license. We do not publish or mirror the gated checkpoint.
