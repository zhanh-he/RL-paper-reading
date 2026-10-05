# Replay visuals

These images are generated from the adjacent public audio assets. Each waveform
uses a linear peak envelope. The spectrograms are mel-frequency power spectra,
not CQT or log-Hz FFT plots. They extend to 24 kHz for 48 kHz audio, or the
source Nyquist frequency when lower (22.05 kHz for 44.1 kHz audio). All use a
fixed black-floor viridis palette and -80 to 0 dB display limits for paired
comparison. The quietest bins and the image background are pure black.

```sh
ffmpeg -hide_banner -loglevel error -y -i "$INPUT" -filter_complex 'aformat=channel_layouts=mono,showwavespic=s=1200x140:colors=0x31bba6:scale=lin:filter=peak' -frames:v 1 "${STEM}_wave.png"
uv run --with librosa --with matplotlib --with pillow python platform/scripts/render-mel-visuals.py
```

All lyrics-to-song images can be regenerated together with
`node platform/scripts/render-lyrics-visuals.mjs`. The waveform filter uses
linear full-scale amplitude (same -1 to +1 bounds for every clip). The mel
spectrogram is labeled with frequency ticks; its dB values are relative to a
full-scale PCM sinusoid and are not reward scores.

The added YuE2 high-LR and canonical MuseCritic waveforms were rendered with
`platform/scripts/render-wave-visuals.py`, which also uses fixed -1 to +1
amplitude bounds and a black background. Their mel images use the same
`render-mel-visuals.py` palette, frequency axis, and dB limits as the earlier
replays.

The figures are visual aids, not additional reward measurements. The synthetic
SATB piano roll on the MIDI reward page is drawn in `lab.js`; it is deliberately
not an experimental prediction.
