# SATB reward audit, 2026-09-29

`receipt.json` summarizes four existing local ChoralStream MIDI prediction/reference pairs and deterministic controlled variants of their references. It is a diagnostic reward audit, not a random held-out benchmark and not an RL result. Raw MIDI and source identifiers stay on lab5090.

Reproduce on the machine that holds the pairs:

```bash
python audit_midi.py --directory /path/to/transcribe_output --output receipt.json
```

The controlled variants test whether a reward detects a missing Bass line, swapped Soprano/Alto, 100 ms note timing shift, fragmented sustained notes, and octave overfill. Use track-aware onset/note matching as primary rewards; active-voice and fragmentation scores are diagnostics or small auxiliary terms, because either alone can miss severe note errors.
