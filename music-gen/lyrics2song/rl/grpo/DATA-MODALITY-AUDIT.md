# Lyrics2song data and input-modality audit

This note distinguishes generation conditioning from preference supervision. It
uses the pinned CMI-Pref and WildSongBench metadata recorded in
[`FORMAL-CMI-PREF.md`](FORMAL-CMI-PREF.md); source audio is not needed for the
metadata counts, and reference-audio assets have **not** been downloaded.

## What the CMI fields mean

The [CMI-RewardBench paper](https://arxiv.org/html/2603.00610v3) defines a
compositional prompt as `(text description, lyrics, reference audio)`, with each
part optional. Its four evaluated strata are text, text+lyrics, text+audio,
and text+audio+lyrics. The [CMI-Pref dataset
card](https://huggingface.co/datasets/HaiwenXia/cmi-pref) sometimes abbreviates
the latter two as "audio-only" and "audio+lyrics"; the released `prompt` field
still carries text. All 469 train votes with both `lyrics` and
`ref-audio-path` have nonempty `prompt`. Thus "audio+lyrics" must **not** be
silently interpreted as an audio-and-lyrics-only comparison.

`ref-audio-path` is the *input* reference recording. `audio-path` and `audio2`
are the two *generated outputs* judged by annotators. Rows are individual
human votes, so repeated prompts/pairs do not become independent conditions.
The feedback and musicality/alignment labels are valuable for reward-model
evaluation or preference learning; the current on-policy GRPO uses only prompt
conditions, not those human preference labels.

| CMI-Pref condition | Official train votes | Official test votes | Current YuE2 run |
| --- | ---: | ---: | --- |
| text | 1,874 | 125 | Excluded: no supplied lyrics |
| text + reference audio | 770 | 125 | Excluded: no lyrics or native audio-reference input |
| text + lyrics | 414 | 125 | Source of the current prompt split |
| text + reference audio + lyrics | 469 | 125 | Separate multimodal task; not merged |

The official test has 500 balanced votes and stays sealed. One test row in the
text+lyrics stratum has an empty `prompt`; all four train strata have nonempty
`prompt`. The paper describes audio references as style-transfer or
continuation conditions, but the released rows do not provide a verified
per-row task label for that distinction.

A provisional metadata-only audit of the 469 text+audio+lyrics train votes
removes 34 whose normalized lyrics match the sealed CMI-Pref test or WSB, then
finds 300 distinct `(style, lyrics, reference-audio path)` conditions among the
remaining 435 votes. This is **not** a frozen train/validation split: the
reference files are not staged, the task semantics have not been checked, and
mixing it with text+lyrics would require grouping shared lyrics before any
split. Nine lyric texts occur in both train modality strata.

## Frozen text+lyrics split

From the 414 eligible train votes: exclude 23 votes whose normalized lyrics
match CMI-Pref official test, exclude zero matching WildSongBench lyrics, and
collapse 98 repeated style+lyrics votes. The 293 remaining unique conditions
are ordered by SHA256 condition ID, with the first 59 assigned to validation
and the other 234 to training. Direct audit of the frozen manifest found no
exact normalized-lyrics or source-prompt-ID overlap between train and
validation. Exact matching does not exclude paraphrases or hidden reuse in
model pretraining. Manifest SHA256:
`c29217883289d3717c510d671927319d5e7acb70ad00443d2a3d9384653ac23b`.

The [WildSongBench card](https://huggingface.co/datasets/m-a-p/WildSongBench)
publishes 192 test prompts (94 Chinese, 98 English) with style, lyrics, and
seeds, **not reference audio**. All 192 are reserved for one final external
test after validation-based model selection. A 600-semantic-token output is a
short-clip test on those prompts, not a reproduction of the benchmark's
full-song protocol. The official CMI-Pref test is an additional sealed
preference benchmark, not the selected WSB generation test.

## Model interface and fair comparisons

- [YuE2's public generation API](https://github.com/multimodal-art-projection/YuE/blob/main/docs/generation.md)
  takes `style`, `lyrics`, and optionally `abc` (plus sampling controls). It
  has no raw `reference_audio` request field. Its documented cover route
  transcribes audio externally with SheetSage2, reviews the ABC score, then
  conditions YuE2 on that score. This is an **audio-to-score bridge**, not
  native audio+lyrics conditioning.
- [Muse's paper](https://arxiv.org/html/2601.03973) and
  [inference guide](https://github.com/yuhui1038/Muse/blob/main/infer/README.md)
  describe global and segment-level text style, structured lyrics, and
  autoregressively generated MuCodec audio tokens. Earlier *generated* audio
  tokens provide context for subsequent segments. The released inference guide
  does not document a way to condition on an arbitrary external reference
  recording. Such an interface should not be assumed from the codec alone.

Consequently, the completed YuE2 SongEval arms are a **text+lyrics baseline**.
They do not test a native audio+lyrics or text+audio+lyrics model. If comparing
against a model whose input includes an audio reference, use two separate
tracks: (A) common text+lyrics input for every model, if that model supports
disabling audio; (B) a reference-audio task on a dataset that actually supplies
reference audio, with native-audio systems clearly separated from YuE2's
external-transcription bridge. WSB alone cannot score track B. Do not compare
track A and B as if they had equal information.

Before track B can be frozen, specify whether the reference is a style cue,
melody/performance guide, accompaniment, or continuation prefix; whether style
text is also supplied; the allowed reference duration; and whether the same
recording is available to every model. The current CMI-Pref train metadata has
469 text+audio+lyrics votes, but its reference-audio files are not staged and
its role labels need a task audit. No new multimodal GRPO arm should be
declared comparable until these questions are settled.
