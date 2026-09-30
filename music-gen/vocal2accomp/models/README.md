# Vocal-conditioned model adapters

Use one folder per integrated checkpoint family. [LaDA-Band](lada-band/README.md) is the preferred dry-vocal-to-accompaniment target; its gated weights and dependencies are staged privately on 5090 and Gadi, but no LaDA inference or GRPO result is yet measured. [ACE-Step 1.5](ace-step-1.5/README.md) and [AnyAccomp](anyaccomp/README.md) are measured baselines with different output contracts. Record input audio format, training support, revision and license before comparisons. Do not commit weights.
