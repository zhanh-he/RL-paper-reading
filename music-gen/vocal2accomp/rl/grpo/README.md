# LaDA-Band online GRPO

`train_lada_band.py` samples two complete masked-diffusion trajectories per update. It records sampled tokens, masked positions, old log probabilities, and the exogenous random-remask schedule; then replays those same actions under the current policy for a group-relative policy-gradient objective. A frozen base output head supplies the reference distribution for the sampled-KL penalty. There is only **one optimizer pass per sampled group**: the probability ratio begins at approximately one, so the written clipping term is effectively inactive, unlike multi-epoch PPO/GRPO. This is best described as single-pass GRPO-style online RL, not a validated multi-epoch GRPO recipe. The 1B backbone, CLaMP3, and MuCodec remain frozen; only a rank-8 LoRA on `to_logits` is trained.

The controlled sampler uses the official top-k/top-p filters plus a 0.01% full-softmax component. This keeps all token actions in support when replaying after an update. Baseline and every checkpoint use the same sampler, text prompt, input vocal and evaluation seed. It is **not** byte-identical to the official low-confidence remask demo: random remasking is used so mask positions do not depend on policy probabilities.

## Measured 4-second smoke, 30 Sep 2026

On the RTX 5090, official LaDA-Band checkpoint `lada_band_lm1B_total3B.ckpt` generated a 4-second accompaniment in 10.28 seconds after loading with eight denoise steps. A separate coverage-only five-update GRPO smoke took 19.1-19.3 seconds per update, group size two, on the original synthetic vowel guide. The LoRA B weight norm after update 5 was 0.0694, so optimization did run. Fixed-seed RMS coverage remained 0.9802 at both steps 0 and 5; no audio-quality improvement is claimed. That reward is nearly saturated on the held-fixed sample and is retained only as a diagnostic arm.

The main presentation run uses one ACE Studio Emma vocal: six seconds for online updates and twelve seconds for fixed-seed playback. The combined *proxy* reward is 0.45 RMS coverage + 0.35 energy-onset fit + 0.20 spectral-band occupancy, with clipping/peak and spectral-flatness penalties. Energy-onset fit is **not** Madmom Beat-v2; band occupancy is **not** a validated perceptual richness model. Each component is logged separately so a reward increase cannot hide a worse waveform. This one-singer demo does not establish out-of-song generalization or human preference.

## Fixed-seed 12-second replay (measured, 30 Sep 2026)

All rows use the same Emma input, prompt, eight-step sampler and evaluation seed 777. Beat-v2 and STFT coverage are the original vocal2accomp offline reward implementations, not the proxies optimized by this run.

| Step | Combined proxy | RMS coverage | Beat-v2 F1 | STFT coverage | Stereo RMS | Acc/vocal RMS dB | Stereo peak | Clipped samples |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.4067 | 0.5282 | 0.2963 | 0.2958 | 0.0194 | -7.2 | 0.2235 | 0 |
| 5 | 0.4067 | 0.5282 | 0.2963 | 0.2958 | 0.0194 | -7.2 | 0.2235 | 0 |
| 50 | 0.4241 | 0.5349 | 0.3571 | 0.2498 | 0.0173 | -8.1 | 0.2230 | 0 |
| 100 | 0.4410 | 0.5615 | 0.5000 | 0.1364 | 0.0140 | -10.0 | 0.1301 | 0 |
| 150 | 0.5964 | 0.8738 | 0.4286 | 0.6484 | 0.0328 | -2.6 | 0.3351 | 0 |
| 200 | 0.5737 | 0.7841 | 0.2069 | 0.4296 | 0.0221 | -6.0 | 0.1952 | 0 |
| 300 | 0.6240 | 0.9169 | 0.5714 | 0.7653 | 0.0587 | +2.5 | 0.6361 | 0 |

The step-5 WAV is byte-identical to baseline. By step 100, the optimized proxy **on the fixed replay** and independent Beat-v2 metric improve, but independent STFT coverage falls by more than half and overall RMS falls by about 28%. At step 200, the onset-fit proxy reaches 0.223 (baseline 0.075), yet the original Beat-v2 F1 falls to 0.207 (baseline 0.296). At step 300, all three listed scores rise, but accompaniment RMS moves from 7.2 dB below the fixed vocal to 2.5 dB above it, a mix-balance warning rather than a listening verdict. These are metric disagreements on one replay, **not** proof that the policy generalizes or that listeners prefer it. The 12-second evaluation includes the six seconds used for training, so it is not a held-out-song test. All measured checkpoints have zero clipped samples; full-scale clipping is not the explanation for these outputs. Listen to the [paired demo](../../../../platform/site/demos/vocal-lada.html) before judging quality.

The band-occupancy proxy is `0.7143` at every fixed 12-second checkpoint, so it does not discriminate among these seven outputs; the combined objective's measured movement here comes from coverage and onset fit. The synthetic attack audit shows that the flatness penalty suppresses white noise, but this run does **not** establish that reward combination generally prevents hacking or loudness drift.

The sampled training-candidate mean reward was 0.517 for steps 1-10 and 0.489 for steps 91-100. Those windows use different sampled trajectories and are not a fixed-seed comparison; they do show that this tiny run has no monotonic on-policy reward increase. The improvement above refers only to the fixed replay.

For that same fixed replay, the fraction of 40 ms frames above the reward's 0.01 RMS threshold rises from about 52.7% to 56.0%, while mean RMS *within those active frames* falls from 0.0214 to 0.0152. Spreading thinner energy across more frames is a plausible threshold-reward failure mode, but the measured waveform and listener judgment are needed before calling it harmful reward hacking.

## Reproduce on lab5090

The official code and gated assets remain outside Git at the paths in [the model note](../../models/lada-band/README.md). Use the isolated `.env` created inside the code checkout. The run should be given exclusive use of the 5090; YuE2 and LaDA peak allocations cannot coexist safely.

```bash
export LADA_PRETRAINED_ROOT=/home/mengh/research/LaDA-Band-assets/pretrained
export LADA_CODE_ROOT=/home/mengh/research/LaDA-Band-posttrain/codes
export HF_HUB_CACHE="$LADA_PRETRAINED_ROOT" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /home/mengh/research/LaDA-Band-posttrain
.env/bin/python train_lada_band.py \
  --code-root codes \
  --checkpoint /home/mengh/research/LaDA-Band-assets/checkpoints/lada_band_lm1B_total3B.ckpt \
  --vocal ace_emma_vocal_16s.wav \
  --output outputs/grpo_emma_combined_6s \
  --seconds 6 --eval-seconds 12 --denoise-steps 8 --group 2 \
  --reward combined --lr 5e-4 --steps 100 --save-steps 5 50 100
```

The completed continuation used the same arguments with `--steps 300 --save-steps 150 200 300 --resume outputs/grpo_emma_combined_6s/step_0100.pt`. The checkpoint includes both adapter and optimizer state. `metrics.jsonl` holds per-group scores; `evaluations.jsonl` contains only fixed-seed replay measurements. The public demo exporter copies WAV/PNG/metric data only, never gated weights.

`evaluate_lada_band.py` replayed all saved adapters on seconds 6-12 of the **same** Emma recording, which were excluded from online updates. It saved a separate source WAV, checkpoint WAVs and metrics without changing weights:

```bash
.env/bin/python evaluate_lada_band.py \
  --run-dir outputs/grpo_emma_combined_6s \
  --output outputs/grpo_emma_combined_6s_heldout_phrase \
  --start-seconds 6 --duration-seconds 6 --steps 0 5 50 100 150 200 300
```

| Step | Combined proxy | Beat-v2 F1 | STFT coverage | Acc/vocal RMS dB | Clipped samples |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.234 | 0.000 | 0.012 | -18.1 | 0 |
| 5 | 0.234 | 0.000 | 0.012 | -18.1 | 0 |
| 50 | 0.234 | 0.000 | 0.012 | -18.1 | 0 |
| 100 | 0.352 | 0.167 | 0.157 | -8.4 | 0 |
| 150 | 0.352 | 0.167 | 0.157 | -8.4 | 0 |
| 200 | 0.552 | 0.333 | 0.847 | -1.4 | 0 |
| 300 | 0.351 | 0.154 | 0.095 | -10.2 | 0 |

On this phrase, 200 to 300 steps is non-monotonic for all three scores, despite an increase on the overlapping 12-second replay. The beat reference contains only six beats, so these F1 values are high-variance. This is a held-out *phrase*, not a held-out song or singer; it cannot establish broad generalization or prove overfitting. The [held-out paired audio](../../../../platform/site/demos/vocal-lada.html?phrase=heldout) is available for listening.

## Coverage-only ablation

The separate adapter starts from the same frozen checkpoint, vocal, prompt, sampler, seeds, group size and learning rate as the combined run. Its only reward is the trainer's **40 ms frame RMS coverage**, not the original vocal2accomp STFT coverage. The completed [coverage-only replay](../../../../platform/site/demos/vocal-lada.html?arm=coverage) and original offline checks are:

| Arm / step | Optimized RMS coverage | Beat-v2 F1 (offline) | STFT coverage (offline) | Acc/vocal RMS dB | Clipped samples |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen baseline / 0 | 0.528 | 0.296 | 0.296 | -7.2 | 0 |
| Combined proxy / 50 | 0.535 | 0.357 | 0.250 | -8.1 | 0 |
| Coverage-only / 50 | 0.944 | 0.500 | 0.802 | +3.4 | 0 |
| Combined proxy / 100 | 0.562 | 0.500 | 0.136 | -10.0 | 0 |
| Coverage-only / 100 | 0.668 | 0.357 | 0.438 | -6.4 | 0 |

The coverage-only arm has much more accompaniment energy than the fixed vocal at 50, but independent beat and STFT-coverage scores also rise, so these metrics alone do not establish poor music or reward hacking. On the same fixed replay, coverage-only falls from 0.944 to 0.668 between steps 50 and 100; Beat-v2 falls 0.500 to 0.357 and STFT coverage 0.802 to 0.438. In 23 of 100 online groups, the two candidates had equal coverage reward; 22 pairs were both at the maximum 1.0. This reward saturation removes the group-relative learning signal for those pairs. The combined arm is quieter at step 50, but its own step-300 loudness drift shows only partial restraint. No listener-preference result is claimed.

```bash
.env/bin/python train_lada_band_v2.py \
  --code-root codes \
  --checkpoint /home/mengh/research/LaDA-Band-assets/checkpoints/lada_band_lm1B_total3B.ckpt \
  --vocal ace_emma_vocal_16s.wav --output outputs/grpo_emma_coverage_6s \
  --seconds 6 --eval-seconds 12 --denoise-steps 8 --group 2 \
  --reward coverage --lr 5e-4 --steps 100 --save-steps 5 50 100
```

## Original Beat-v2 reward arm

The `--reward beat_v2` arm sends each generated candidate WAV to the persistent [`beat_v2_worker.py`](../../rewards/beat_v2_worker.py) process in the existing `auto-beat-reward` environment. That worker calls the original vocal2accomp `MadmomBeatV2Scorer`, with the same fixed vocal and duration-specific reference cache. A CPU preflight returned Beat-v2 F1 `0.1429` for the first six seconds of the frozen baseline and `0.2963` for twelve seconds; the latter matches the independent offline receipt. Online rewards use six seconds, while fixed checkpoint replays use twelve seconds, so those F1 values are not interchangeable. The saved 0/5/50/100/150/200/300-step [Beat-v2 paired replay](../../../../platform/site/demos/vocal-lada.html?arm=beat_v2) is measured:

| Step | Beat-v2 F1, fixed 12 s | RMS coverage, fixed 12 s | STFT coverage, fixed 12 s | Acc/vocal RMS dB | Clipped samples |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.296 | 0.528 | 0.296 | -7.2 | 0 |
| 5 | 0.296 | 0.528 | 0.296 | -7.2 | 0 |
| 50 | 0.216 | 0.880 | 0.641 | -4.8 | 0 |
| 100 | 0.400 | 0.744 | 0.446 | -6.5 | 0 |
| 150 | 0.000 | 0.997 | 0.901 | +1.8 | 0 |
| 200 | 0.000 | 0.970 | 0.824 | +1.1 | 0 |
| 300 | 0.154 | 0.468 | 0.205 | -9.0 | 0 |

The step-5 WAV is byte-identical to baseline. Mean six-second training-candidate F1 was 0.172 in steps 1-10, 0.256 in steps 91-100, 0.358 in steps 191-200, and 0.238 in steps 291-300, but those are different sampled trajectories, not controlled before/after pairs. The 12-second fixed replay falls at step 50, rises to 0.400 at step 100, then reaches 0 at steps 150 and 200 despite 13 detected accompaniment beats and high coverage in both. At step 300 F1 recovers only to 0.154, below the frozen baseline's 0.296; STFT coverage also falls to 0.205. Of 300 online groups, 22 had equal Beat-v2 scores, versus 23 equal pairs in the first 100 coverage-only groups. These trajectories show metric/segment disagreement and reward-specific trade-offs, not a verified listener-quality direction. The separate [constructed click attack and actual sparse online candidate](../../rewards/README.md) are labeled and available for listening; neither establishes that training caused a hack. The step-150 adapter/optimizer and evaluation were saved before a coordinated GPU handoff; continuation to 300 resumed from that full checkpoint and completed.

```bash
.env/bin/python train_lada_band_v2.py \
  --code-root codes \
  --checkpoint /home/mengh/research/LaDA-Band-assets/checkpoints/lada_band_lm1B_total3B.ckpt \
  --vocal ace_emma_vocal_16s.wav --output outputs/grpo_emma_beat_v2_6s \
  --seconds 6 --eval-seconds 12 --denoise-steps 8 --group 2 \
  --reward beat_v2 --lr 5e-4 --steps 100 --save-steps 5 50 100 \
  --beat-worker-python /home/mengh/miniconda3/envs/auto-beat-reward/bin/python \
  --beat-worker-script beat_v2_worker.py \
  --beat-reward-root /home/mengh/research/vocal2accomp-muse
```

The persistent worker scores actual generated audio, not the fast onset-fit proxy. An unscorable vocal reference yields zero reward, and the record keeps reference/accompaniment beat counts. This is still a single-pass group-relative update with the same limited LoRA scope and one fixed training vocal.

## Calibrated two-component guard arm (50-step interim)

The additional `beat_v2_coverage_guard` option uses **actual** Madmom Beat-v2 F1 `B` and 40 ms RMS coverage `C`, with a coverage gate on beat credit and a coverage term that stops growing after 0.7:

`R = 0.65 B min(1, C/0.4) + 0.35 min(1, C/0.7) - Q - 0.025 D - 0.5 max(0, flatness-0.3)`

Here `Q` is the existing peak/clipping/flatness quality penalty and `D` is the sum of dB distance outside an accompaniment-to-vocal RMS window of `[-18, -3] dB`. The gate is designed to suppress beat-rich near-silence; saturation removes the incentive to increase loudness once coverage is adequate. This remains a **partial** combination: there is no validated richness reward, and no claim that the guard prevents all gaming. The pure function passed six unit tests on lab5090; for the oracle-aligned 6-second click track with Beat-v2 F1 `1.0`, it returns `-0.561` instead of a high score. That is a function stress test, not a trained-model result.

The matched GPU arm is running with the same frozen baseline, input, prompt, sampler, group size, learning rate and seeds. Its step-0 WAV has the same SHA-256 across all four arms. The 0/5/50-step [interim paired replay](../../../../platform/site/demos/vocal-lada.html?arm=guarded) is independently scored:

| Step | Guarded reward | Beat-v2 F1 | RMS coverage | STFT coverage | Acc/vocal RMS dB | Clipped samples |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.457 | 0.296 | 0.528 | 0.296 | -7.2 | 0 |
| 5 | 0.457 | 0.296 | 0.528 | 0.296 | -7.2 | 0 |
| 50 | 0.491 | 0.216 | 0.947 | 0.730 | -3.9 | 0 |

At step 50, the combined training objective rises but Beat-v2 F1 falls below baseline. Higher coverage and STFT coverage are not proof of better arrangement, and this interim result **does not establish that combination prevents reward hacking**. The 100-step checkpoint and held-out phrase remain pending. Run command:

```bash
.env/bin/python train_lada_band_guarded.py \
  --code-root codes \
  --checkpoint /home/mengh/research/LaDA-Band-assets/checkpoints/lada_band_lm1B_total3B.ckpt \
  --vocal ace_emma_vocal_16s.wav --output outputs/grpo_emma_guarded_6s \
  --seconds 6 --eval-seconds 12 --denoise-steps 8 --group 2 \
  --reward beat_v2_coverage_guard --lr 5e-4 --steps 100 --save-steps 5 50 100 \
  --beat-worker-python /home/mengh/miniconda3/envs/auto-beat-reward/bin/python \
  --beat-worker-script beat_v2_worker.py \
  --beat-reward-root /home/mengh/research/vocal2accomp-muse
```
