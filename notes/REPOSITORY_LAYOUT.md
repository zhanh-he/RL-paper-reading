# Repository layout

```text
data/literature.csv          Shared literature source of truth
notes/                       Public synthesis and experiment-record template
site/index.html              Existing literature website; URL stays stable
site/demos/                  One self-contained static demo per verified run
site/overview/               Optional future cross-experiment results website
update-request/literature/   One paper/catalog request per file
update-request/new-methods/ One method proposal per file
music-trans/
  multi-inst/
  choral-singing/
music-gen/
  vocal2accomp/
  lyrics2song/
```

Each of the four task directories uses the same ownership split:

```text
<task>/
  README.md                  Task contract, datasets, metrics and status
  models/<model-slug>/       Adapter/config and upstream provenance; no weights
  rewards/                   Independent reward implementations and tests
  rl/dpo/<run-id>/           Pair construction, config, receipts and evaluation
  rl/grpo/<run-id>/          Rollout, reward, policy-update receipts and evaluation
```

Use lowercase kebab-case for folders and snake_case for Python modules. Do not create a new folder for a one-file reward: `vocal2accomp/rewards/beat.py`, `coverage.py` and `richness.py` are the intended endpoints. `combine.py` belongs beside them and should only compose calibrated scores with documented guardrails. A reward folder becomes appropriate only when a reward actually needs multiple source files, tests or assets. The `rl/` directories organize methods, not reward definitions; a new optimizer such as OPSD can be added as `rl/opsd/` when there is a real run.

The public README is the overview now. Once several verified experiments have comparable metrics, `site/overview/` can become a results website. Each run's demo gets a stable `site/demos/<task>/<run-id>/` URL and links back to its record. The website never substitutes for a record of model revision, data license, split, reward version, budget and independent evaluation.

This repository is public. Internal data, model weights and restricted audio stay outside Git; only approved, redistributable examples may enter `site/assets/` or a demo. Existing root `Update_request_*.txt` files are legacy pending inputs and are intentionally untouched by this migration.
