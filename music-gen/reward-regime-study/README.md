# SP / AI Reward Regimes

[Public experiments](https://zhanh-he.github.io/RL-paper-reading/demos/#reward-regimes).
The versioned [protocol](protocol.json) was **approved for six formal training
arms on 2026-10-14**, one learning rate **3e-4**, seed42, 270 training steps.
[Progress](progress.json) contains checkpoint-verified traces and only complete
30-condition validation means, with receipt hashes. No synthetic curves or
invented MOS values are presented as results.

The approved core compares original YuE2 and Muse bases under spectral Richness,
SongBench-Instrument, and SongBench-Arrangement. The user confirmed Arrangement
as the third reward on 2026-10-14 and subsequently approved training. SongEval
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

## Execution

- YuE2 Richness: lab5090 PID3839302. Instrument then Arrangement follow in a
  finite Bash queue PID3842776, sharing exact frozen baseline audio.
- Muse Richness: Gadi wa66 job181223774; Instrument: iv96 job181223777;
  Arrangement: wa66 job181223779. All submitted priority1023, 1 Hopper GPU,
  8-hour walltime. Submitted does not mean running.
  Each arm shares its80GB GPU between actor, native decoder and serialized
  reward/rollout services; this reduces the allocation requirement. Actual
  startup memory compatibility still requires a compute-node run.
- Old owned 5090/controller and Mac transport processes are SIGSTOP-paused;
  old Gadi jobs are user-held and 74 owned pending Kaya evaluation jobs held.
  Results/checkpoints preserved. No unrelated LaDA/CASM tasks interrupted.
- No scheduled Codex tasks were created. The finite queue executes actual
  training jobs; it is not a reminder or future chat automation.

The scorer is serialized, moving frozen Demucs and SongBench onto GPU only for
scoring. It scores private PCM24-FLAC and records SHA256, all seven SongBench
dimensions, SP components, raw levels and common penalties. Public listening
copies must be MP3. Guards are explicit in `protocol.json` and `rewards.py`.
Learning objectives are separate single-target arms, not a combined reward.
KL beta0.01 remains a separate loss term; KL traces are sampled training
estimates, not exact all-token-distribution divergences.

YuE2 uses PyTorch/PEFT, Muse uses native ms-swift/TRL + vLLM. Muse's study-specific
multi-turn scheduler fixes512 tokens/turn; top-k filtering is disabled explicitly
on both actor and rollout server. Validation uses all30 frozen prompts and
seeds5101+index. YuE2 checkpoints0/50/100/270; Muse0/270. Current short validation
rollouts are **not** the separately requested30-60-second listening demos.
After all three YuE2 arms finish, the same finite queue runs
`render_yue2_replays.py`: three fixed validation examples, shared baseline and
steps50/100/270 per arm, native750-1500 semantic-token duration constraints.
These are explicitly separate from full validation and cannot be called a
benchmark mean. No loops/padding or loudness normalization; output MP3 hashes
are distinct from the scored/source lossless originals.

Collect aggregate progress with `collect_progress.py --root STUDY_OUTPUT_ROOT`
using the YuE2 Python environment. Build/check the site from `platform` after
recovery. This collector exports no raw private prompts, audio or checkpoints.
