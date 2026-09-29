# Repository layout

```text
AGENTS.md                    Agent protocol
README.md                    Team entrypoint and experiment overview
upd_request/
  Update_request_*.txt       Three stable contributor request files
  done_requests/             Completed request archive
platform/
  data/literature.csv        Shared literature source of truth
  docs/                      Catalog editing and maintainer documentation
  scripts/                   Catalog build/validation
  site/index.html            Existing literature website; URL stays stable
  site/demos/                Static pages for verified runs
  package.json               Local build and preview commands
notes/                       Public synthesis and experiment-record template
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

The public README is the overview now. Once several verified experiments have comparable metrics, `platform/site/overview/` can become a results website. Each run's demo gets a stable deployed `/demos/<task>/<run-id>/` URL and links back to its record. The website never substitutes for a record of model revision, data license, split, reward version, budget and independent evaluation.

This repository is public. Internal data, model weights and restricted audio stay outside Git; only approved, redistributable examples may enter `platform/site/assets/` or a demo. The three request texts and completed archives were moved under `upd_request/` without processing their contents. Hidden root files such as `.github/`, `.gitignore`, and `.pages.yml` remain where GitHub tooling requires them.
