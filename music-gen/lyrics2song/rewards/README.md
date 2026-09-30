# Lyrics-to-song rewards

The implemented YuE2/SongEval arm is documented in
[its exact reward and optimization protocol](songeval-grpo-protocol.md).
SongEval five-score mean is the only reward in that arm. The
[47-clip perturbation audit](audits/2026-09-29/README.md) probes gain and
clipping separately; it is not an RL result.

Run separate arms from the same initial checkpoint: SongEval aesthetics, MuseCritic Mean5, CMI-RM alignment, CMI-RM musicality/quality and an independent audio-aesthetic scorer. Record each scorer's checkpoint and preprocessing version. CMI-RewardBench is an evaluation benchmark, not the callable training reward. Keep lyric intelligibility/adherence and blinded listening outside the optimized score; check silence, truncation, repetition and style/genre shortcuts.

Add one small module per integrated scorer (`songeval.py`, `musecritic.py`, `cmi_rm.py`, `audio_aesthetics.py`) with tests and output diagnostics. Only add `combine.py` after single-reward ablations and calibration. No scorer implementation is claimed here.
