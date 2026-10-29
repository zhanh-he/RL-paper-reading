# AMT live experiments

Live page: <https://zhanh-he.github.io/RL-paper-reading/amt/demos/>.
The AMT catalog's Experiments button opens it; the MusicGen live page also links to it.
The page follows the MusicGen live layout: experiment/sample selection, compact Reward/KL traces,
checkpoint selection, input/reference/baseline/post-training replay, checkpoint metrics,
then separate Dataset Protocol, Models and Rewards tabs.

Frozen reliability audits and genuine CPU metric controls are registered separately from training. Historical
[Choral GRPO](https://zhanh-he.github.io/RL-paper-reading/demos/#choral) and
[SATB reward audits](https://zhanh-he.github.io/RL-paper-reading/demos/#choral-rewards)
retain their original URLs and receipts. Do not reinterpret them as a new AMT experiment.

## Files and ownership

- Edit `platform/data/amt/experiments.json` for public experiment records.
- `platform/site/amt/demos/data/experiments.json` is a generated copy.
- Page code lives in `platform/site/amt/demos/{index.html,app.js,styles.css}`.
- The validator/publisher is `platform/scripts/sync-amt-experiments.mjs`.
- Full run records and experiment code belong under `music-trans/<task>/rl/<method>/runs/<run-id>/`.
- AMT literature remains in `platform/data/amt/literature.csv`; no literature changes are required to update replay assets.

Use `npm --prefix platform run build`, `npm --prefix platform run check` and
`npm --prefix platform run test:amt`. The build rejects missing assets, non-MP3 audio,
invalid MIDI notes, results on planned checkpoints, and measured stages without receipts.
For browser checks, run `npm --prefix platform run test:amt:ui` with Playwright and
its dedicated testing Chromium. `PLAYWRIGHT_MODULE` can point at the shared runtime's
`playwright/index.mjs`. The UI test injects its fixture in memory; no test results are published.

## Manifest contract

Top-level keys are `schema_version: 1`, `updated_at: YYYY-MM-DD`, `experiments`,
`datasets`, `models` and `rewards`. All four collections are arrays. A minimal planned
record can be added only after an experiment has been chosen:

```json
{
  "id": "chosen-experiment",
  "title": "Chosen experiment title",
  "status": "planned",
  "task": "Chosen AMT task",
  "model": "Chosen model and version",
  "method": "Chosen training method",
  "reward": "Chosen objective",
  "samples": [],
  "stages": [],
  "trace": []
}
```

Statuses are `planned`, `queued`, `running`, `measured`, `failed`. `id` is stable lowercase ASCII with
hyphens; `?run=<id>` opens a particular experiment. Tab deep links are `#experiments`,
`#datasets`, `#models`, `#rewards`.

For a run with results, add a repository-relative `receipt` path and an
`evaluation_protocol` describing matching tolerances, track semantics, frame grid,
aggregation and test identity. Do not train or choose checkpoints from a sealed test.

| Field | Shape / meaning |
| --- | --- |
| `samples` | `{id, title, duration, input?, reference?, note?}`; duration in seconds. Input/reference are assets, independently optional. |
| Asset | `{audio?, midi?, notes?, tracks?}`. Audio and MIDI use site-root-relative paths such as `demos/audio/…mp3` or `amt/demos/midi/…mid`; paths have no leading slash. |
| `notes` | `[{track, pitch, start, end}]`. Track is a zero-based integer; pitch is MIDI 0–127; times are seconds within the sample. Empty notes explicitly mean a silent prediction. |
| `tracks` | Optional array of track names, e.g. Soprano/Alto/Tenor/Bass or instrument names. Every output for a sample must use the same track index/name/color mapping. Never relabel an instrument track as SATB without an adapter. |
| `stages` | Sorted unique `[{step, status, outputs?, evaluation?}]`, beginning with frozen baseline step 0. Steps count optimizer updates. Missing outputs stay empty rather than borrowing another checkpoint. |
| `outputs` | Object keyed by sample ID; each value is an asset. A public output requires a measured stage and a run receipt. |
| `evaluation` | `{cohort, n, frame?, onset?, onset_offset?}`. F1 values are fractions in [0,1]; UI shows one decimal percent. Deltas are percentage points, shown only for matching cohort ID and N. |
| `trace` | Ordered `[{step, reward?, kl?}]`. Missing metrics stay absent. Values must be finite. |
| `trace_source` / `trace_note` | Required for traces: repository-relative log/receipt and its measurement scope (online/offline KL, reference policy, averaging/window, etc.). |

Training curves show the nearest **logged** value on hover and at a selected checkpoint;
they do not interpolate a measured value for a missing step. All MIDI views for a sample
share pitch/time limits and colors, and open a larger piano roll on click. Audio playback
uses the supplied file; no automatic timbre or loudness changes are applied by this page.
Record scoring-original and public MP3 hashes, render settings, matched singers/voices
and paired decode settings in the run receipt. A replay sample and cohort mean have
separate roles and must be described in `sample.note` and `evaluation_protocol`.

## Frozen evaluation runs

Use `kind:"evaluation"` with `stages:[]` and `trace:[]`; never encode perturbations
as optimizer steps or invent Reward/KL curves. `conditions` contains unique
`{id,label,status,outputs?,evaluation?,error?}` records. Status may also be `queued`
or `failed`; failures require a receipt and error text. A measured condition needs
a receipt and genuine outputs or scores. Unmeasured conditions cannot contain
results. A queued snapshot means submitted, not GPU execution.

`metric_definitions` declares `{id,label,unit}` with `unit:"fraction"` or `"number"`.
Evaluation is `{cohort,n,values:{metricId:value}}`; fractions are in [0,1], whereas
Pearson r may be negative and cents error is an unbounded numeric quantity.
Artificial metric controls, procedural model smoke, real vocal benchmarks and
generation comparisons must be separate runs with their own evidence scope.

Optional `reported_results` is reserved for the fixed SheetSage2 v1 Table 17
reference comparison: `{name,rwc,rock,output,source}`. Its two F1 values are
percentages in [0,100], not local evaluation fractions. The Models tab labels
them author-reported, with dataset, protocol, source and date; never convert
them into a measured local experiment or combine different-task scores there.

Dataset rows use `{name, labels, split, source, source_label?}`. Model rows use
`{name, interface, checkpoint, source, source_label?}`. Reward rows use
`{name, definition, failure_case, source, source_label?}`. Sources use HTTPS links to
the paper, official release or this repository's versioned experiment receipt. List the
actual objective and scope, not a claimed improvement without evidence.

Publish after build/check and an isolated-browser smoke check. Preserve the historical
demo and the two literature catalogs. This page setup does not authorize starting training.
