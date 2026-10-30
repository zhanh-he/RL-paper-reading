# Score excerpts and model outputs

Source: ASAP (Foscarin et al., ISMIR 2020), via the
[pinned dataset fork](https://github.com/TimFelixBeyer/asap-dataset/tree/8cba199e15931975542010a7ea2ff94a6fc9cbee).
Dataset material and these reference excerpts/adaptations are CC BY-NC-SA 4.0;
see the [source license](https://github.com/TimFelixBeyer/asap-dataset/blob/8cba199e15931975542010a7ea2ff94a6fc9cbee/LICENSE.md).

- `test-02`: Bach, Prelude BWV 891; performance BLINOV04M.
- `test-07`: Beethoven, Piano Sonata 10, movement 1; performance Hou02M.

Reference MusicXML is copied from that benchmark. Baseline, SFT50 and GRPO50 MusicXML
are actual DuET model outputs from the same performance. Displayed SVGs are the
first four measures, rendered by Verovio 6.3.0; they are excerpts, not the full-score
evaluation. The full MusicXML files remain downloadable. Excerpt cut points may
leave ties/slurs open. No audio recording is included.

GRPO50 denotes 50 rollout batches / 40 effective duration-head updates, not 50
full-model optimizer steps. Both SFT and GRPO use the scope in the run receipt.

The two works were selected after evaluation to illustrate one improved and one
worsened complete-score error, not sampled to estimate an average. The result
table uses all eight preselected test works. See the
[run receipt](../../../../../music-trans/score-bench/runs/2026-10-30/README.md).
