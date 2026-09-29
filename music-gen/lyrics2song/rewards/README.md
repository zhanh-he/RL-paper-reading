# Lyrics-to-song rewards

Run separate arms from the same initial checkpoint: SongEval aesthetics, MuseCritic Mean5, CMI-RM alignment, CMI-RM musicality/quality and an independent audio-aesthetic scorer. Record each scorer's checkpoint and preprocessing version. CMI-RewardBench is an evaluation benchmark, not the callable training reward. Keep lyric intelligibility/adherence and blinded listening outside the optimized score; check silence, truncation, repetition and style/genre shortcuts.

Add one small module per integrated scorer (`songeval.py`, `musecritic.py`, `cmi_rm.py`, `audio_aesthetics.py`) with tests and output diagnostics. Only add `combine.py` after single-reward ablations and calibration. No scorer implementation is claimed here.
