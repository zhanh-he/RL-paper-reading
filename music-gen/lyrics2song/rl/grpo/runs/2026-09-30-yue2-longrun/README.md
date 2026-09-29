# YuE2-3B SongEval-GRPO, 100 optimizer updates

This local academic run used the released YuE2-3B and YuE2-VAE weights on
lab5090. [Training code](../../yue2_songeval_longrun.py) contains all 8
original training prompts and 3 disjoint original held-out prompts. It used
2 on-policy rollouts per update, SongEval five-score mean as reward, a LoRA
adapter, learning rate `2e-5`, fixed held-out seeds 5101/5102/5103, and a
600-semantic-token cap. The latter may truncate musical structure; each
receipt records truncation. The AdamW state was reset when this run resumed
from the previous day's one-step LoRA.

| Optimizer update | 0 | 1 | 5 | 50 | 100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mean SongEval, 3 held-out prompts | 3.8403 | 3.6030 | 3.6160 | 3.5704 | 3.6063 |

The 0-update output is the frozen YuE2 base in this evaluation configuration.
The 1-update stage replays the **source adapter** (SHA-256
`8dee8838d2c565bc6ffe777b8c95471e5e00cf3ca754e86d88cb1155242bd597`);
updates 2-100 continue from it. The update-100 adapter SHA-256 is
`0e51b6f470bb7b3fb129735f23861fa735dceec2d831dab876d357ec5f819983`.
The five milestone receipts are in `step_*/receipt.json`;
the [public replay](https://zhanh-he.github.io/RL-paper-reading/demos/#lyrics)
offers one matched held-out audio example per milestone with waveform,
spectrogram and signal diagnostics.

At 100 updates, the three held-out peaks were 0.8512, 0.8165 and 0.8167,
with zero samples at or above 0.999. The held-out mean remains below the
0-update baseline, so this run provides neither a quality-gain claim nor an
example of pervasive clipping. The three prompts and one training seed are
too small to infer general behavior; independent listening is still needed.
