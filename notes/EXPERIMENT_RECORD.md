# Experiment record template

Copy this into `rl/<method>/<run-id>/README.md` for a real run. Delete unused fields; do not fill planned values with predicted outcomes.

| Field | Value |
| --- | --- |
| Status | planned / running / verified / failed |
| Evidence | local measurement / published comparison |
| Task and model | upstream URL, revision, size, license and starting checkpoint |
| Method | DPO / GRPO / other, implementation revision |
| Reward | exact module/version, scale, aggregation and guardrails |
| Data | source, license, train/dev/test split, source-level de-duplication |
| Budget | samples, candidate K, reward calls, updates, GPU-hours, seeds |
| Controls | base, same-budget SFT, Best-of-K, random reward as applicable |
| Metrics | primary endpoint, independent checks, uncertainty method |
| Artifacts | config, logs, tables, licensed audio/MIDI demo URL |

## Results

State baseline and post-training values with units and confidence intervals. Keep train reward, held-out proxy, independent automatic metrics and human evaluation separate. Mark unpublished or incomplete cells explicitly; a generated example is not evidence of a training gain.

## Failure review

Record reward hacking, silence/repetition, wrong instruments or voices, timing drift, data leakage, and license limitations. Link to the exact code and artifact revision used for every displayed figure.
