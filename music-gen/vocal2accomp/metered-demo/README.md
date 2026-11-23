# Accompaniment Lab controls

The four control tracks are original procedural signals. Two grooves preserve the reference downbeats. A third delays the backing at bar 5, mutes bar 6 and returns on bar 7. The explicit beat markers test whether the interface can show a timing fault and a valid alternative groove. The detector is an amplitude probe for these designed signals.

From the root of `RL-paper-reading`, rebuild the control assets with Python 3.10+ and ffmpeg.

```sh
python3 music-gen/vocal2accomp/metered-demo/make_fixtures.py --output platform/site/demos/vocal-accompaniment/audio
python3 music-gen/vocal2accomp/metered-demo/analyze_fixtures.py --manifest platform/site/demos/vocal-accompaniment/audio/manifest.json
```

The AnyAccomp session replays a separate published inference. Its vocal guide has note anchors and no annotated meter. The original audio bindings and run settings are in the session provenance file. Browser judgments are local and the page starts with zero judgments on a new origin.
