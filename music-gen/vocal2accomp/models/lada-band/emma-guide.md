# Fixed Emma vocal for LaDA-Band replay

This is an original, ACE Studio Vocal Synth rendering of the nine-note melody in [`guide_vocal.py`](../ace-step-1.5/guide_vocal.py). The voice is Emma (V2), with `la` on every note, at 120 BPM. A single Sing track has no instrument or accompaniment. The 16-second, 48 kHz mono WAV was exported with external effects disabled; the public [12-second input](../../../../platform/site/demos/audio/ace_emma_vocal_12s.wav) is a direct PCM trim. It is generated singing, **not** a human recording.

The editable ACE project and untrimmed export remain locally at `/Users/jollibear/Documents/Codex/2026-07-03/wo/work/ACE-Vocal2Accomp-Emma/`. The public demo uses the first 6 seconds for online updates and the first 12 seconds for fixed-seed before/after playback. No paired ground-truth accompaniment exists. All generated outputs should be evaluated against the same input clip and prompt, not against this vocal as a full mix.
