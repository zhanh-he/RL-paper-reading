# Early single-reward checkpoints

Each `arm_*_1.json` is an anonymous held-out aggregate from a separate
one-step GRPO rerun. Each `arm_*_100.json` comes from step 100 of the original
300-step run for that arm. All runs start from the same frozen ChoralStream
checkpoint, seed 29 and 16-train/8-test-song split. The five 300-step head files
from those original runs are byte-identical to step 300 of the 1000-step runs.

`public-synthetic-replay-{1,100}.json` scores one original synthetic song,
which is not in the held-out set. `public-synthetic-notes.json` describes its
MIDI geometry. The model checkpoint, training songs and per-song test results
remain off the public repository. ACE Studio audio is for listening only;
it uses the same four singers, `la` lyrics and inherited template language as
the 300-step renders. Objective F1 is computed from the unmodified MIDI.
