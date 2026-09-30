# LaDA-Band (Tencent Music Entertainment)

Official [paper](https://arxiv.org/abs/2604.11052), [code](https://github.com/Duoluoluos/LaDA-Band), and [gated weights](https://huggingface.co/sDuoluoluos/LaDA-Band). This is the preferred vocal-to-accompaniment candidate: it takes a dry vocal and optional text and generates an accompaniment, rather than replacing the singer or outputting a full mixed song.

## Verified local status, 30 Sep 2026

- The official code has been cloned to lab5090 at `/home/mengh/research/LaDA-Band-posttrain`. It contains `codes/infer.py`, `codes/train.py`, the model implementation and training configs.
- The main checkpoint is `checkpoints/lada_band_lm1B_total3B.ckpt` in the authors' manually reviewed Hugging Face repository. A direct download from the current lab5090 account returned HTTP 403. No checkpoint, baseline audio or GRPO adapter has been obtained or run.
- The official quick-start requires additional Llama-3.2-1B, MERT, CLaMP3 and MuCodec assets and CUDA. The public Git repository alone is not an executable pretrained model.

## Post-training path

LaDA-Band uses discrete masked diffusion over accompaniment codec tokens with vocal-prefix conditioning. The released `train.py` is a supervised Lightning entry point, not a GRPO trainer. A valid online RL implementation must sample multiple complete accompaniment trajectories for the same fixed vocal, retain the selected masked-token transition log-probabilities, compute group-relative advantages, and replay those exact transitions under updated and frozen policies. A reward computed on a final waveform must be attached to the full denoising trajectory. Do not substitute autoregressive token likelihood or pretend the released training entry point performs GRPO.

Once authorized weights are present, first verify a 16-second fixed-guide inference under the official config, then adapt `coverage`, `beat-v2` and `richness` as individually logged rewards with an audio-quality/length guardrail. Keep vocal input, random seed, sampling schedule and renderer fixed for baseline and each post-training checkpoint. Until the inference baseline is measurable, this folder makes no LaDA-Band quality claim.
