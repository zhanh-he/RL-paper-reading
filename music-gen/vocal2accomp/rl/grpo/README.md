# LaDA-Band online GRPO

`train_lada_band.py` samples two complete masked-diffusion trajectories per update. It records sampled tokens, masked positions, old log probabilities, and the exogenous random-remask schedule; then replays those same actions under the current policy for a clipped group-relative objective. A frozen base output head supplies the reference distribution for the sampled-KL penalty. The 1B backbone, CLaMP3, and MuCodec remain frozen; only a rank-8 LoRA on `to_logits` is trained. This is a head-adapter experiment, not full-model GRPO.

The controlled sampler uses the official top-k/top-p filters plus a 0.01% full-softmax component. This keeps all token actions in support when replaying after an update. Baseline and every checkpoint use the same sampler, text prompt, input vocal and evaluation seed. It is **not** byte-identical to the official low-confidence remask demo: random remasking is used so mask positions do not depend on policy probabilities.

## Measured 4-second smoke, 30 Sep 2026

On the RTX 5090, official LaDA-Band checkpoint `lada_band_lm1B_total3B.ckpt` generated a 4-second accompaniment in 10.28 seconds after loading with eight denoise steps. A separate coverage-only five-update GRPO smoke took 19.1-19.3 seconds per update, group size two, on the original synthetic vowel guide. The LoRA B weight norm after update 5 was 0.0694, so optimization did run. Fixed-seed RMS coverage remained 0.9802 at both steps 0 and 5; no audio-quality improvement is claimed. That reward is nearly saturated on the held-fixed sample and is retained only as a diagnostic arm.

The main presentation run uses one ACE Studio Emma vocal: six seconds for online updates and twelve seconds for fixed-seed playback. The combined *proxy* reward is 0.45 RMS coverage + 0.35 energy-onset fit + 0.20 spectral-band occupancy, with clipping/peak and spectral-flatness penalties. Energy-onset fit is **not** Madmom Beat-v2; band occupancy is **not** a validated perceptual richness model. Each component is logged separately so a reward increase cannot hide a worse waveform. This one-singer demo does not establish out-of-song generalization or human preference.

## Fixed-seed 12-second replay (measured, 30 Sep 2026)

All rows use the same Emma input, prompt, eight-step sampler and evaluation seed 777. Beat-v2 and STFT coverage are the original vocal2accomp offline reward implementations, not the proxies optimized by this run.

| Step | Combined proxy | RMS coverage | Beat-v2 F1 | STFT coverage | Stereo RMS | Stereo peak | Clipped samples |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.4067 | 0.5282 | 0.2963 | 0.2958 | 0.0194 | 0.2235 | 0 |
| 5 | 0.4067 | 0.5282 | 0.2963 | 0.2958 | 0.0194 | 0.2235 | 0 |
| 50 | 0.4241 | 0.5349 | 0.3571 | 0.2498 | 0.0173 | 0.2230 | 0 |
| 100 | 0.4410 | 0.5615 | 0.5000 | 0.1364 | 0.0140 | 0.1301 | 0 |

The step-5 WAV is byte-identical to baseline. By step 100, the optimized proxy and independent Beat-v2 metric improve, but independent STFT coverage falls by more than half and overall RMS falls by about 28%. This is evidence of a metric trade-off on one replay, **not** proof that the policy generalizes or that listeners prefer it. The 12-second evaluation includes the six seconds used for training, so it is not a held-out-song test. The lower peak and zero clipping also rule out full-scale clipping as the explanation for this particular output. Listen to the [paired demo](../../../../platform/site/demos/vocal-lada.html) before judging quality. The 300-step checkpoint is pending GPU rotation with YuE2.

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

Resume to 300 with the same arguments, `--steps 300 --save-steps 300 --resume outputs/grpo_emma_combined_6s/step_0100.pt`. The checkpoint includes both adapter and optimizer state. `metrics.jsonl` holds per-group scores; `evaluations.jsonl` contains only fixed-seed replay measurements. The public demo exporter copies WAV/PNG/metric data only, never gated weights.

After the 300-step checkpoint exists, `evaluate_lada_band.py` can replay all saved adapters on seconds 6-12 of the **same** Emma recording, which were excluded from online updates. It saves a separate source WAV, checkpoint WAVs and metrics without changing weights:

```bash
.env/bin/python evaluate_lada_band.py \
  --run-dir outputs/grpo_emma_combined_6s \
  --output outputs/grpo_emma_combined_6s_heldout_phrase \
  --start-seconds 6 --duration-seconds 6 --steps 0 5 50 100 300
```

This is a held-out *phrase*, not a held-out song or singer; it cannot establish broad generalization.
