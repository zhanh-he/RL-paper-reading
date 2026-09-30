# GRPO runs

The [MuScriptor-medium token-head smoke](runs/2026-09-30-muscriptor-medium-smoke/README.md)
uses a previously authorized local checkpoint. It implements group-relative
advantages, clipped token likelihood ratios and frozen-reference KL. The
single training song and 8-song short-excerpt audit are exploratory, not an
instrument-aware Multi F1 benchmark. Future runs should report the candidate
count, sampling policy, per-candidate reward, group variance, actual KL,
effective updates and independent full-song Multi F1.
