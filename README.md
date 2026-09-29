# Music Post-Training Lab

[Literature catalog](https://zhanh-he.github.io/RL-paper-reading/) · [Experiment map](#experiment-map) · [Request an update](upd_request/) · [Editing guide](platform/docs/EDITING.md)

This repository keeps the shared RL literature catalog and the public-facing structure for music post-training experiments. The GitHub Pages URL remains the literature site; its source, data and build tools now live together under [`platform/`](platform/). This README is the experiment overview until verified results justify a separate overview site.

## Experiment Map

| Area | Task | Current state |
| --- | --- | --- |
| [Music transcription](music-trans/README.md) | [Multi-instrument](music-trans/multi-inst/README.md) | Model, reward and DPO/GRPO layout; no local training result published here |
| [Music transcription](music-trans/README.md) | [Choral singing](music-trans/choral-singing/README.md) | SATB-aware reward plan; no local training result published here |
| [Music generation](music-gen/README.md) | [Fixed vocal to accompaniment](music-gen/vocal2accomp/README.md) | Beat, coverage, richness and combined reward contracts |
| [Music generation](music-gen/README.md) | [Lyrics to song](music-gen/lyrics2song/README.md) | Open-model and reward comparison plan |

Each task owns `models/`, `rewards/` and `rl/{dpo,grpo}/`. A model folder holds an adapter/config and provenance, not weights. Simple reward implementations belong in one Python file each; `combine.py` at the reward root composes them after their scales and failure modes have been checked. A run gets its own folder under the appropriate RL method, with a reproducible record using the [experiment template](notes/EXPERIMENT_RECORD.md). Public demo pages will live under [`platform/site/demos/`](platform/site/demos/README.md) when actual, redistributable artifacts exist.

## Literature And Updates

1. 在 [interactive catalog](https://zhanh-he.github.io/RL-paper-reading/) 搜索、筛选和排序文献。
2. 修改请求写进 [`upd_request/`](upd_request/) 里自己的 TXT：[Zhanh](upd_request/Update_request_zhanh.txt)、[Felix](upd_request/Update_request_felix.txt)、[Hanyu](upd_request/Update_request_hanyu.txt)。完成的请求归档在 [`done_requests/`](upd_request/done_requests/)；这次搬目录没有处理待办内容。新实验方法直接归入所属任务的 `rl/`，不再维护第二套请求目录。
3. 团队可以直接编辑 [`platform/data/literature.csv`](platform/data/literature.csv)。`platform/site/data/literature.json` 是生成文件，不要手改；具体字段见 [editing guide](platform/docs/EDITING.md)。

The repository is public. Do not commit private Obsidian notes, restricted datasets, model checkpoints, credentials or audio without redistribution rights. Published paper results, local measurements and proposed experiments must be labeled separately.

## Labels
- Domain：`MusicGen / MusicEval / MIR / AudioGen / Speech / SpeechEnhance / AudioLLM / LLM / CV / MachineLearning / SourceSep / Multimodal / Other`
- Workstream：`Reward / RL / Reward-n-RL / Other`
- Zhanh 与 Felix 的星级分别由本人维护；`*` 表示评分存疑、等待重读。三人的解读颜色为 Zhanh 深红、Felix 绿色、Hanyu 蓝色。
