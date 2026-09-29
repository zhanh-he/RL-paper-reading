# Demo audio

The two WAVs are an original synthetic sung-vowel guide (not a human singer)
and a single ACE-Step 1.5 base-model completion. Neither is vocal-to-accompaniment
GRPO nor an isolated accompaniment track.

The YuE2 and Muse FLAC pairs are generated from original English lyrics with
the released models under noncommercial academic use. `before` and `after`
share a prompt and random seed within each model. Their one-step SongEval-GRPO
receipts and no-update replay checks live in
`music-gen/lyrics2song/rl/grpo/runs/2026-09-29-{yue2,muse}/`. These short
examples must not be represented as human-rated full songs or as evidence of
any proprietary system's behavior. FLAC is lossless; no source music or model
weights are redistributed.
