# SheetSage2 raw-audio audit

Submitted Slurm job: **248098**. Scope: one V100-32GB, four CPU cores, 64 GB RAM,
30-minute limit, no more than 48 inference attempts. Other queued/held jobs were
not modified. No training or codec bootstrap is performed.

Inputs: 12 original procedural harmonic probes x original / 500ms leading silence /
-6dB gain / 30dB SNR noise. These are related diagnostic cases, not independent
musical works and not singing. They cannot establish real vocal accuracy.

Model and parent assets are existing pinned releases. No checkpoints are published.
The 300-second model window is unchanged. V100 has no native BF16, so the worker
uses FP32 when required and explicitly records this departure from paper precision.
Keep load failures, OOM, parse/export failures and incomplete attempts visible.

This receipt initially records submission, not completion. The machine state and
subsequent measured results or failure receipt must be attached before any score
is presented. The public manifest may lag a running remote process until a new
verified snapshot is published; it is not a live scheduler connection.

Scheduler snapshot checked 2026-10-29, approximately 15:15 AWST: PENDING (Priority).
The scheduler estimated 2026-10-30 01:45:41 start; estimates can change and are
not a promise. No allocated GPU or measured inference score existed in this snapshot.
