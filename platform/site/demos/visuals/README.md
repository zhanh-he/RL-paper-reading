# Replay visuals

These images are generated from the adjacent public audio assets. Each waveform
uses a linear peak envelope; each spectrogram uses the same 40 Hz-16 kHz log
frequency range and 80 dB display range so a paired image is comparable.

```sh
ffmpeg -hide_banner -loglevel error -y -i "$INPUT" -filter_complex 'aformat=channel_layouts=mono,showwavespic=s=1200x140:colors=0x31bba6:scale=lin:filter=peak' -frames:v 1 "${STEM}_wave.png"
ffmpeg -hide_banner -loglevel error -y -i "$INPUT" -filter_complex 'aformat=channel_layouts=mono,showspectrumpic=s=1200x320:legend=0:scale=log:fscale=log:start=40:stop=16000:color=viridis:drange=80:limit=0' -frames:v 1 "${STEM}_spectrum.png"
```

The figures are visual aids, not additional reward measurements. The synthetic
SATB piano roll on the MIDI reward page is drawn in `lab.js`; it is deliberately
not an experimental prediction.
