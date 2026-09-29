# Muse SongEval GRPO one-step pilot

**Status:** measured on-policy engineering pilot, not a quality claim. Released
[Muse-0.6b](https://github.com/yuhui1038/Muse) weights sampled two original
English lyric/style fragments (500 generated tokens each), which the released
[MuCodec](https://github.com/tencent-ailab/MuCodec) weights decoded at 48 kHz
stereo with 20 diffusion steps. The official SongEval five-score mean supplied
group-relative advantages for one attention q/v LoRA GRPO update.

| Held-out prompt, seed 5101, 19.8 s each | SongEval mean | Peak | Samples >= 0.999 |
| --- | ---: | ---: | ---: |
| Before | 3.5585 | 0.717 | 0% |
| After | 3.3330 | 0.495 | 0% |

The distinct held-out prompt uses different original lyrics/style. A no-update
replay reproduced the exact decoded audio samples. The one-step LoRA update had
nonzero gradient and changed 347/500 generated token positions. It did **not**
produce full-scale clipping, but held-out SongEval decreased. A prior
unconstrained pilot stopped early after training (19.8 to 11.2 seconds), so
this published comparison forces 500 generated tokens on both sides. The two
training candidates were both about 19.7 seconds, though they still differed
in loudness and content; the higher-reward candidate was much louder.

The [receipt](receipt.json), [policy pilot](../../muse_songeval_pilot.py) and
[MuCodec decoder](../../decode_muse_tokens.py) document the measured run.
Listen in the [experiment demo](../../../../../../platform/site/demos/index.html#lyrics).
One group, one update and one held-out prompt cannot determine long-run GRPO
behavior, human quality, or whether SongEval causes any clipping in another
model. These noncommercial academic examples use original text; the public
site does not redistribute model weights or source music.
