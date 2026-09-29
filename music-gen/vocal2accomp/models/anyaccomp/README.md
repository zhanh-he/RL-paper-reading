# AnyAccomp

Upstream: [official code](https://github.com/AmphionTeam/AnyAccomp) and [weights](https://huggingface.co/amphion/anyaccomp). This model generates accompaniment conditioned on vocal or solo-instrument melody.

The [inference receipt](receipt.json) records one real lab5090 run with the same 16-second synthetic vocal guide used for ACE-Step 1.5. The [demo](https://zhanh-he.github.io/RL-paper-reading/demos/#vocal) lets listeners compare the vocal input, isolated AnyAccomp accompaniment, its vocal-plus-accompaniment mixture and ACE-Step's full-mix completion. Both are baseline inference only. The accompaniment-only output should not be ranked against the full-mix output without listening to the mixture with the fixed original vocal. No GRPO or reward result is claimed.
