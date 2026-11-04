# Independent SongEval check

This experiment uses the same 25 WAV files as Audiobox/PAM. All file SHA256 values were checked across the current manifest, WAV files, both raw local scorer JSONL files, the joined local results and SongEval results. No human labels or new RL training are present.

The existing Linux CPU environment ran Python 3.12.11. Its relevant exact packages are in `requirements-songeval.txt`. Neither that environment nor the shared SongEval checkout was modified. The first attempt to import the original evaluation helper lacked Hydra; an already installed dependency directory resolved that import. The final minimal runner directly instantiates the official `model.Generator` using the official YAML config and strict checkpoint key matching, so it does not require Hydra.

For a clean environment, install the listed packages, clone the official repository and use the pinned backbone:

```bash
git clone https://github.com/ASLP-lab/SongEval.git SongEval
git -C SongEval checkout 848fb2ff3a2a9d64dcb20d46f07238c28abd7add
python -m pip install -r requirements-songeval.txt
python -c 'from huggingface_hub import snapshot_download; snapshot_download("OpenMuQ/MuQ-large-msd-iter", revision="0562a57814f6f8bbd9fdea0a25921a2fce1a841a", local_dir="muq-pinned")'
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' python score_songeval.py --manifest songeval_manifest.json --audio-dir scoring_audio --songeval SongEval --muq muq-pinned --output songeval_results.json --threads 4
```

The runtime manifest needs `samples:[{id,excerpt_sha256}]` and the audio directory must contain each `id.wav`. `scoring_audio` contains private lossless scoring copies; the public repository distributes MP3 listening copies only. Those MP3s are not exact replacements for the scored WAVs.

The run uses `librosa.load(...,sr=24000,mono=True)` exactly as official `eval.py`, then MuQ `hidden_states[6]`, and the official five-score head. It stores unrounded raw predictions. The official CLI rounds the same outputs to four decimals. No gain normalization occurs.

Expected weight SHA256:

- SongEval head: `0cb1eb8be02b96e9949ca3a2b66997c5ac7d9da3567240df4dbfc2048eb98efd`
- MuQ: `273febab2be02872c37d2c37e48a9d6c52c1c9392f3eeeabd498efa281ccb7a6`

The CPU was an AMD Ryzen 9 7900. This additional measurement uses different hardware and a different timing scope from the M4 Audiobox/PAM run, so its speed should not form a direct leaderboard. SongEval is a song-aesthetics diagnostic. Naturalness targets vocal breathing and phrasing; Clarity targets song structure. A ten-second excerpt cannot establish full-song structure or memorability, and these outputs do not supply human artifact labels.

Primary sources: [official SongEval code](https://github.com/ASLP-lab/SongEval), [paper](https://arxiv.org/abs/2505.10793), [MuQ weights](https://huggingface.co/OpenMuQ/MuQ-large-msd-iter/tree/0562a57814f6f8bbd9fdea0a25921a2fce1a841a). SongEval README licenses the toolkit CC-BY-NC-SA4.0; the MuQ backbone weights carry CC-BY-NC4.0.
