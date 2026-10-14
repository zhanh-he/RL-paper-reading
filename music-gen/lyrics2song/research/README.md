# Lyrics2song research map

Verified against public papers, official repositories and model/dataset cards on
2026-10-13. This is a research reference, not a leaderboard or a claim about
field-wide usage frequency. No new model training is launched by this update.

## Online Reference

- [Source-checked post-training reading map](2026-10-14-post-training-reading.md):
  LeVo 2, YuE2 and Full-Song equations/results, version differences, OPD
  foundations and teacher-side distribution shift. Paper evidence stays separate
  from our experiment receipts.
- [Models](https://zhanh-he.github.io/RL-paper-reading/demos/#lyrics-models):
  eight model families/releases, native inputs, code vs weights vs training
  recipes, separate licenses, and our assessment of post-training feasibility.
- [Rewards](https://zhanh-he.github.io/RL-paper-reading/demos/#lyrics-rewards):
  nine reward/evaluation categories, actual definitions, evidence and failure
  modes. Proposals are not retroactively added to measured objectives.
- [Dataset Protocol](https://zhanh-he.github.io/RL-paper-reading/demos/#datasets):
  Lyrics2song is the default; eight public data resources and a collapsible
  comparison of official protocols with our frozen CMI-Pref split.

The editable source is [catalog.json](catalog.json). Each entry carries direct
primary-source links. `platform/scripts/sync-lyrics-reference.mjs` validates and
publishes it to `platform/site/demos/data/lyrics-reference.json`; the public JSON
is downloadable from all three reference panels. Do not manually edit that copy.
The 2026-10-14 reading pass also updates shared literature metadata and adds two
OPD references. Contributor ratings are unchanged.

## Decisions And Boundaries

1. Keep YuE2 and Muse as current experimental policies. Their data, codec and
   architecture differ: they are not a clean parameter-count intervention.
2. Next inference/adaptation candidates: LeVo 2 and HeartMuLa. Native audio
   conditioning needs its own comparison group: SongGen's reference voice,
   LeVo's reference style and YuE2's audio-to-score cover pipeline do different
   things. HeartMuLa 3B's ordinary released interface is tags plus lyrics;
   MuLaCover is a separate release. Do not equate metadata completeness with
   three-modal conditioning in every policy.
3. A released AR inference model may be adapted for token-policy GRPO, but this
   is our engineering assessment, not proof of an official GRPO implementation.
   Flow/DiT updates require a valid trajectory objective; optimizing only their
   LM planner is a separate experiment. SFT/LoRA availability is not GRPO support.
4. SongEval and MuseCritic share the five-aspect SongEval rubric. SongEval scores
   audio features directly; MuseCritic generates critiques before scalar heads.
   Their raw scores and score changes are not interchangeable. Shared supervision
   also means they are not completely independent perceptual judges.
5. CMI-RewardBench is the framework, CMI-Pref the human preference data, CMI-RM
   the scorer. The latter's quality/alignment outputs are useful candidates but
   need a training-data overlap audit before serving as an independent evaluator.
6. SongBloom's historical official release is currently inaccessible. Keep it
   as a paper reference; forks do not establish present official weight access
   or license. Original unavailable URLs are retained in the source JSON.

## Frozen Experiment Protocol

The [formal protocol](../rl/grpo/FORMAL-CMI-PREF.md) is unchanged:
270 training conditions, 30 validation conditions, 121 sealed official-test
conditions (125 votes) from the three-modal-source subset of CMI-Pref. The
[downloadable split](https://zhanh-he.github.io/RL-paper-reading/demos/data/cmi-pref-triple-9to1-split.json)
contains identifiers/hashes, not republished lyrics or recordings.

The 9:1 split is **ours**, not an official validation split. We group conditions
by normalized lyrics after isolating official-test lyrics. Current YuE2/Muse
policy inputs are text plus lyrics. GRPO uses the conditions for fresh online
rollouts, not the released A/B preference labels. A preference vote is not an
independent song and neither candidate is a unique ground-truth waveform.

Public dataset cards evolve: freeze revisions and hashes when reproducing them.
For example, the current CMI-Pref-Pseudo card reports 165k comparisons, of which
115k are pseudo-labeled; an earlier paper count must not replace release counts.
Muse's paper reserves 256 songs for validation and 100 for test; actual IDs must
be verified before claiming split reproduction. WildSongBench's standard uses
two candidates with PER selection and a specific decoder/version: our fixed
single-seed replay is not that benchmark protocol.

## Proposed Evaluation And Ablations

- Separate online reward/reference-policy KL, fixed validation selection and
  final sealed-test metrics. Embedding/distribution KL is not policy KL.
- Keep prompt adapters, seeds, candidate budget, decoding controls and codec
  fixed. Account for failed generations; do not silently filter them out.
- Report aesthetic axes, lyric PER/WER/CER, style alignment and signal failures
  separately. ASR errors and shared reward-model data are limitations.
- Compare single objectives, calibrated combinations without guards, and
  combinations with guards. First test gain, clipping, silence, noise,
  repetition and omitted-lyric attacks; never assume added metrics stop hacking.
- Longer replay is not longer validation. The current approximately 64-second
  public inference remains distinct from 600-token training/validation rollouts.
  A new long-form test should report 60-second and naturally completed-song
  results, including omissions, repeats, duration and truncation.
- Listening copies remain MP3 only, 192 kbps / 48 kHz, without normalization.
  Private lossless scoring originals and public MP3s have separate SHA256s.
  A separate loudness-matched listening study must be labeled as such.

These are proposed additional comparisons, not newly completed experiments.

## Verification

```bash
npm --prefix platform run build
npm --prefix platform run check
DEMO_URL=http://127.0.0.1:4173/demos/ node platform/scripts/test-lyrics-reference.mjs
```

Start `npm --prefix platform run serve` for the browser test. It checks
source-backed row counts, architecture filters, task
switches, legacy Vocal links, download metadata, default Lyrics dataset view,
desktop/mobile overflow, and JavaScript errors. Playwright runtime/channel and
the demo URL can be supplied through the usual environment variables.
