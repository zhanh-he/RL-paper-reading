# LaDA-Band (Tencent Music Entertainment)

Official [paper](https://arxiv.org/abs/2604.11052), [code](https://github.com/Duoluoluos/LaDA-Band), and [gated weights](https://huggingface.co/sDuoluoluos/LaDA-Band). This is the preferred vocal-to-accompaniment candidate: it takes a dry vocal and optional text and generates an accompaniment, rather than replacing the singer or outputting a full mixed song.

## Verified local status, 30 Sep 2026

- Official code is on lab5090 at `/home/mengh/research/LaDA-Band-posttrain` and Gadi at `/scratch/wa66/hm6416/LaDA-Band-posttrain`; both check out commit `1177fafd43a5f9b5166aa938c52f73180fdff46f`.
- An approved account's read token was supplied to each download process through standard input, without saving a token on either host. The complete Hugging Face snapshot at revision `6d444caee85385677b0652ecb0b2b8220436dd37` is now at `/home/mengh/research/LaDA-Band-assets` and `/scratch/wa66/hm6416/LaDA-Band-assets`. Each copy has 56 real files, approximately 43 GiB on disk, including the Llama, MERT, CLaMP3 and MuCodec dependencies. The relative-path/file-size inventories match.
- Both copies of `checkpoints/lada_band_lm1B_total3B.ckpt` are 20,695,860,213 bytes and have SHA-256 `050ee4e4c67fd7e06947f475d21af3a13e183344ce790e06a79076a148ef82c5`. Keep these gated files outside Git. The previous 403 applied to a different Hugging Face account and no longer blocks this authorized local staging.
- Baseline inference and GRPO have **not** yet run. The dedicated inference environment, fixed-vocal smoke, and a diffusion-trajectory policy adapter still need to be validated. Gadi's wa66 scratch copy is intended for H200 jobs under either wa66 or iv96; Kaya's V100s are not the target for the BF16 configuration.

## Post-training path

LaDA-Band uses discrete masked diffusion over accompaniment codec tokens with vocal-prefix conditioning. The released `train.py` is a supervised Lightning entry point, not a GRPO trainer. A valid online RL implementation must sample multiple complete accompaniment trajectories for the same fixed vocal, retain the selected masked-token transition log-probabilities, compute group-relative advantages, and replay those exact transitions under updated and frozen policies. A reward computed on a final waveform must be attached to the full denoising trajectory. Do not substitute autoregressive token likelihood or pretend the released training entry point performs GRPO.

With the weights staged, the next gate is a 16-second fixed-guide inference under the official config. Then adapt `coverage`, `beat-v2` and `richness` as individually logged rewards with an audio-quality/length guardrail. Keep vocal input, random seed, sampling schedule and renderer fixed for baseline and each post-training checkpoint. Until the inference baseline is measurable, this folder makes no LaDA-Band quality claim.
