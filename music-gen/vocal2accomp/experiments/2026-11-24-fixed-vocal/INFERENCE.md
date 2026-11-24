# Reproduce the fixed-vocal inference

Use the original authorized WAV and matching lyric transcript for each song. Keep inputs, lossless outputs, inference logs, and full receipts in the private experiment directory. The repository contains the runnable protocol and public evidence.

## AnyAccomp and LaDA-Band on CUDA

Set up the upstream environment and weight layout using the project guidance. The results use the commits below and a GeForce RTX 5090.

| Engine | Upstream commit | Weight revision | Conditioning |
| --- | --- | --- | --- |
| AnyAccomp | `82604b5e3107944ad4c49fc64900b86118ae2c62` | `9aa9e62427337bf1df4caa3c4f3e6ad934522e711` | Dry vocal |
| LaDA-Band | `1177fafd43a5f9b5166aa938c52f73180fdff46f` | `6d444caee85385677b0652ecb0b2b8220436dd37` | Dry vocal and arrangement description |

AnyAccomp expects `pretrained/flow_matching`, `pretrained/vocoder`, and `pretrained/vq` beneath its checkout. The runner calls its pipeline and retains the generated sample count. It uses 50 steps and CFG 3.0. [Upstream code](https://github.com/AmphionTeam/AnyAccomp) · [Weights](https://huggingface.co/amphion/anyaccomp)

LaDA-Band uses the 1B inference configuration, 24 denoising steps, and a reused arrangement instruction with a style-specific instrument description. The upstream direct-vocal entry point reports the effective per-song seed derived from the supplied seed and input basename. Use the same basename to reproduce that seed. The asset directory needs both `pretrained/` and `checkpoints/`. The offline Hugging Face cache defaults to `pretrained/` and can be supplied with `--hf-cache` if the layout differs. [Upstream code](https://github.com/Duoluoluos/LaDA-Band) · [Gated research weights](https://huggingface.co/sDuoluoluos/LaDA-Band)

Run these commands with the selected upstream environment activated.

```bash
CUDA_VISIBLE_DEVICES=0 python run_open_models.py anyaccomp \
  "$PRIVATE/vocadito_33.wav" "$PRIVATE/anyaccomp-seed29" \
  --code-root "$ANYACCOMP_CHECKOUT" --style unconditioned --seed 29

CUDA_VISIBLE_DEVICES=0 python run_open_models.py lada-band \
  "$PRIVATE/vocadito_33.wav" "$PRIVATE/lada-folk-seed29" \
  --code-root "$LADA_CHECKOUT" --assets-root "$LADA_ASSETS" \
  --style acoustic-folk --seed 29 --lada-steps 24
```

The styled matrix uses `acoustic-folk`, `sparse-synth-pop`, and `half-time-rnb` with seeds `29` and `73` per source. Each run writes `backing.wav`, `mix.wav`, and `receipt.json` under its private candidate directory. The mix sums input and backing at time zero with fixed 0.5 gains, using the original dry sample rate. The receipt records the measured input and output hashes, checkpoint verification, generation time, and peak PyTorch CUDA allocator memory. The measured lab batch reuses its model load for the remaining candidates, which affects its reported wall times.

## StepAudio 3 Music smoke test

Use `--dry-run` to validate the input and write a preview locally. Use `--submit` when an authorized `STEP_API_KEY` is already set in the execution environment. The task uses the original file bytes as raw Base64 and disables both rewrites. It polls once about every five seconds and saves the generated stereo or mono audio exactly as returned for the later duration and lyric review. See the preview model status and the ABC input availability in the model guide. [Model guide](https://platform.stepfun.com/docs/zh/guides/models/stepaudio-3-music) · [Official API reference](https://platform.stepfun.com/docs/zh/api-reference/audio/music.md)

```bash
python stepfun_smoke.py --dry-run \
  --vocal "$PRIVATE/vocadito_33.wav" \
  --lyrics-file "$PRIVATE/vocadito_33_lyrics.txt" \
  --caption "Acoustic folk. Arrange around the supplied sung performance and follow its pickup, timing, held notes and section changes." \
  --private-output "$PRIVATE/stepfun-dry-run"
```

For a live smoke run, use `--submit` with a new private output directory. The private directory contains `receipt.json`, the request body, submit response, last query metadata, and `generated-mix.wav` on success. Confirm the sung words and duration on the returned audio before publication. Waveform-level vocal preservation needs a separate measurement.
