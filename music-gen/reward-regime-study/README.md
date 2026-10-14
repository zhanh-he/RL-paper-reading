# SP / AI Reward Regimes

[Public experiment proposal](https://zhanh-he.github.io/RL-paper-reading/demos/#reward-regimes).
The versioned [protocol](protocol.json) is **not approved for training**. No runs,
synthetic curves or invented MOS values are presented as results.

The proposed core compares original YuE2 and Muse bases under spectral Richness,
SongBench-Instrument, and SongBench-Arrangement. The user confirmed Arrangement
as the third reward on 2026-10-14; this does not authorize training. SongEval
remains independent cross-evaluation, not an optimization arm. A controlled weaker
checkpoint is preferable for isolating baseline quality, but needs independent
train/valid listening, a weight hash and user approval. None of the quality bands
or a MOS 4+ boundary has been established by our matched human evaluation.

[SongEval](https://github.com/ASLP-lab/SongEval) has no instrument head.
[SongBench](https://github.com/Tencent/SongBench) Instrument concerns instrument
realism and synthesis fidelity; Arrangement is the separate orchestration axis.
Neither directly measures instrument count. The existing SP proxy measures
spectral-band activity/movement and tonal flatness, not human MOS. Lyrics2song
needs an explicitly versioned adaptation because it has no fixed vocal input.

[ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) is not justified as a poor
model by its authors' report. Its flow/diffusion generator also needs a different
policy-likelihood implementation from AR-token GRPO. [SongGen](https://github.com/LiuZH-19/SongGen)
has open AR training code/weights but officially supports English <=30 seconds;
silently restricting the full multilingual dataset would invalidate the shared
protocol. These are alternatives, not models with assigned MOS values.
For a distinct older-model candidate, consider [original YuE](https://github.com/multimodal-art-projection/YuE/tree/YuE-v1)
with its released stage-1 weights. Its quality tier still requires matched CMI
train/valid blind listening, and its two-stage codec/log-prob pipeline needs a
separate GRPO adapter and memory assessment. It is proposed, not launched.

Use all frozen 270 CMI train / 30 valid conditions. Reserve all 192 WSB conditions
for this new study until selection and analysis are frozen. Existing WSB
evaluation belongs to a separate campaign and cannot be reused to choose a
reward, model, learning rate or checkpoint for this study. Quality thresholds are
hypotheses: blind listening and reward-family x baseline-quality interactions,
with paired condition-clustered intervals, are required to test them.
