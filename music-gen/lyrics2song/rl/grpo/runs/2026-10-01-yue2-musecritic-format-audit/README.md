# MuseCritic reward-format sensitivity

The original YuE2 MuseCritic trainer scored an intermediate float WAV and
converted the same audio to PCM24 FLAC for replay. A deterministic replay of
held-out prompt 0, seed 5101, reproduced the archived FLAC **byte for byte**
(SHA-256 `42cf99c500f0abd282519b1296ffac6371b8929cfe86e4707dc4b2fefc64f05d`).
Decoded WAV and FLAC differed by at most `5.96e-8` per sample, with RMS
difference `3.48e-8`. Yet the official MuseCritic model, scored in one process
with the same prompt and settings, returned:

| Input | MuseCritic five-score mean |
| --- | ---: |
| Float WAV | 2.196875 |
| PCM24 FLAC | 3.450000 |

The exact FLAC scored 3.450000 in two further independent processes, and
twice identically within the 36-clip cross-scoring batch. The critique text
also changed between WAV and FLAC. The official scorer first generates a
critique and then applies its reward head to the generated sequence; that
architecture is a plausible amplification path, not a proven root cause.
The result demonstrates **format sensitivity on this clip**, not that every
MuseCritic assessment is unstable.

Because the original trainer optimized float-WAV scores while archiving FLAC,
its reward table cannot be interpreted as a stable score for the replayed
audio. `yue2_songeval_longrun.py` now converts to PCM24 FLAC *before* a
MuseCritic reward call and keeps those exact bytes for replay, with SHA-256
in each held-out receipt. `run_yue2_musecritic_pcm24.sh` launches the separate
corrected run, preserving the original as an audit trail.
