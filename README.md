# RL Learning

[MusicGen / RL / Reward catalog](https://zhanh-he.github.io/RL-paper-reading/) · [AMT catalog](https://zhanh-he.github.io/RL-paper-reading/amt/) · [MusicGen Live](https://zhanh-he.github.io/RL-paper-reading/musicgen/) · [Music-RL Live](https://zhanh-he.github.io/RL-paper-reading/demos/) · [AMT live](https://zhanh-he.github.io/RL-paper-reading/amt/demos/) · [Request an update](upd_request/README.md)

One repository, two independent literature catalogs. MusicGen / RL / Reward covers music generation, evaluation, post-training and related methods. AMT covers automatic music transcription, multipitch estimation and voice assignment. Each has its own table, filters, export and update requests. The repository URL stays unchanged so existing links still work.

[MusicGen Live](https://zhanh-he.github.io/RL-paper-reading/musicgen/) tracks music-generation methodology: explicit score planning, AR / NAR, supervised baselines, controllability and editing. It separates research plans, engineering checks, measured training and scientific conclusions. See the [MusicGen publishing guide](platform/docs/MUSICGEN_EXPERIMENTS.md).

The existing [Music-RL Live](https://zhanh-he.github.io/RL-paper-reading/demos/) focuses on post-training, rewards and GRPO, keeping its audio and MIDI replays with versioned receipts. It labels local measurements, reward diagnostics, baseline inference and unfinished GRPO work separately. Its `/demos/` URL and experiment links stay unchanged.

The independent [AMT live page](https://zhanh-he.github.io/RL-paper-reading/amt/demos/) separates measured CPU diagnostic controls, the submitted SheetSage2 GPU audit and planned real-vocal / melody-control experiments. Frozen evaluations use condition-level metrics without invented Reward/KL curves; training replay remains supported. Historical Choral replay stays linked. See the [AMT publishing guide](platform/docs/AMT_EXPERIMENTS.md).

## Quick Prompts

把下面对应的一条发给任意聊天窗，再补上具体论文、修改要求或运行记录。

```text
更新 zhanh-he/RL-paper-reading 的 RL catalog（文献目录）：按 AGENTS.md 和 platform/docs/EDITING.md 处理 upd_request/rl/ 及我本次补充的文献，维护 platform/data/rl/literature.csv，同步相关 Obsidian，构建检查并发布。目标网页是仓库 Pages 根目录。
```

```text
更新 zhanh-he/RL-paper-reading 的 AMT catalog（转录文献目录）：按 AGENTS.md 和 platform/docs/EDITING.md 处理 upd_request/amt/ 及我本次补充的文献，维护 platform/data/amt/literature.csv，同步相关 Obsidian，构建检查并发布。目标网页是 /amt/。
```

```text
更新 zhanh-he/RL-paper-reading 的 MusicGen Live（生成方法论实验页）：先读 AGENTS.md 和 platform/docs/MUSICGEN_EXPERIMENTS.md，按本次要求及已审核的脱敏运行摘要，维护 platform/data/musicgen/experiments.json 与 platform/site/musicgen/，区分计划、工程检查、训练实测和科学结论，构建检查并发布 /musicgen/。不复制私有代码、路径、原始数据或日志；训练需我另行指示。
```

```text
更新 zhanh-he/RL-paper-reading 的 RL experiment webpage（Music-RL Live）：按本次要求和实际运行回执，更新 platform/site/demos/ 及其对应数据源，核对曲线、配对试听和指标后发布 /demos/。仅更新展示；训练需我另行指示。
```

```text
更新 zhanh-he/RL-paper-reading 的 AMT experiment webpage（Music Transcription live）：先读 platform/docs/AMT_EXPERIMENTS.md，按本次要求和实际运行回执维护 platform/data/amt/experiments.json 与 platform/site/amt/demos/，构建检查并发布 /amt/demos/。仅更新展示；训练需我另行指示。
```

## Experiment Map

| Area | Task | Current state |
| --- | --- | --- |
| [Music transcription](music-trans/README.md) | [Multi-instrument](music-trans/multi-inst/README.md) | [MuScriptor hosted baseline](music-trans/multi-inst/models/muscriptor/README.md) on the public synthetic SATB example; separately, a local [medium GRPO feasibility run](music-trans/multi-inst/rl/grpo/runs/2026-09-30-muscriptor-medium-smoke/README.md) has a one-song training replay and exploratory 8-song disjoint short-excerpt audit, not SATB or full-song evidence |
| [Music transcription](music-trans/README.md) | [Choral singing](music-trans/choral-singing/README.md) | [Event-head GRPO](music-trans/choral-singing/rl/grpo/runs/2026-09-30-event-head/README.md) at 0/100/300/1000 steps on 16 train and 8 held-out songs, with actual single-reward ablations, SATB metrics and fixed-singer ACE Studio audio/MIDI replay; [dataset/model/reward research](music-trans/choral-singing/research/README.md) and [earlier frame-head negative control](music-trans/choral-singing/rl/grpo/runs/2026-09-29-frame-head/README.md) stay separate |
| [Music generation](music-gen/README.md) | [Fixed vocal to accompaniment](music-gen/vocal2accomp/README.md) | [Vocal2song GRPO replay](https://zhanh-he.github.io/RL-paper-reading/demos/#vocal) includes LaDA-Band audio, reward ablations and training traces; per-run receipts distinguish measured stages and unavailable outputs |
| [Music generation](music-gen/README.md) | [Lyrics to song](music-gen/lyrics2song/README.md) | Six YuE2 arms completed 270 steps with fixed validation and long MP3 replay. [Muse recovery](music-gen/lyrics2song/rl/grpo/runs/2026-10-14-formal-recovery/README.md): five aligned-LR arms completed 270 steps; MuseCritic-3e-4 resumes from 180 after a decoder repair. Independent Muse inference remains paused. [Live replay](https://zhanh-he.github.io/RL-paper-reading/demos/#lyrics) separates training reward/KL, validation and full-cohort test progress; partial test means are not final results |
| [MusicGen methodology](https://zhanh-he.github.io/RL-paper-reading/musicgen/) | Explicit planning / AR + NAR / controlled generation | Research plan and scoped engineering evidence; music-quality and CoT gains require matched-budget training and evaluation. This page is separate from the existing RL results. |
| [SP / AI reward study](music-gen/reward-regime-study/README.md) | [Proposed quality-regime comparison](https://zhanh-he.github.io/RL-paper-reading/demos/#reward-regimes) | YuE2 / Muse with spectral Richness and learned reward candidates; CMI train/valid, WSB reserved test. Proposal only, awaiting user approval; no invented MOS labels or training results |

Each task owns `models/`, `rewards/` and `rl/{dpo,grpo}/`. A model folder holds an adapter/config and provenance, not weights. Simple reward implementations belong in one Python file each; `combine.py` at the reward root composes them after their scales and failure modes have been checked. A run gets its own folder under the appropriate RL method, with a reproducible record using the [experiment template](notes/EXPERIMENT_RECORD.md). [Model and data selection](notes/2026-09-29-model-data-reward-selection.md) records gates, licenses and recommended next experiments.

## Literature And Updates

1. 选择 [MusicGen / RL / Reward](https://zhanh-he.github.io/RL-paper-reading/) 或 [AMT](https://zhanh-he.github.io/RL-paper-reading/amt/)，搜索、筛选、排序或导出。页面顶部可以切换目录。
2. 修改请求按领域放到 [`upd_request/rl/`](upd_request/rl/) 或 [`upd_request/amt/`](upd_request/amt/)，各有 Zhanh、Other（Felix / Jiahao / 同事）、Hanyu 三个 TXT。[直接编辑入口](upd_request/README.md)不需要安装 GitHub App；标明推荐人及评分后提交即可。完成后按领域带时间戳归档，旧历史不动。
3. 直接维护数据时，编辑 [RL CSV](platform/data/rl/literature.csv) 或 [AMT CSV](platform/data/amt/literature.csv)，不要手改生成的网页和 JSON；具体字段见 [editing guide](platform/docs/EDITING.md)。同一篇文章只保留一个主目录归属，ID、评分、个人解读和原文链接不能因迁移丢失。

The repository is public. Do not commit private Obsidian notes, restricted datasets, model checkpoints, credentials or audio without redistribution rights. Published paper results, local measurements and proposed experiments must be labeled separately.

Public audio is MP3 only, including GitHub release assets: 192 kbps / 48 kHz, without loudness normalization. Keep lossless scoring originals outside the repository; record their hashes separately from MP3 listening copies. Build checks reject other audio formats.

## Labels
- Domain：`MusicGen(LLM) / MusicGen(Small) / MusicGen(Mid) / MusicGen(Large) / MusicGen / MusicFoundation / MusicEval / MusicTranscription / MIR / AudioGen / Speech / SpeechEnhance / AudioLLM / LLM / CV / MachineLearning / SourceSep / Multimodal / Other`
- Workstream：`Reward / RL / Reward-n-RL / Other`
- `MusicFoundation` 留给 MERT/MuQ 等理解表征；`MusicGen(LLM)` 是语言模型/LM+渲染体系。非 LM 生成器按已核验参考配置分 Small <1B、Mid 1–3B、Large >3B；未知规模保留 MusicGen。这是目录约定，不是统一行业标准。
- Domain 下拉框可多选，取消勾选即排除；筛选链接和 CSV 导出保留当前选择。
- “规模 / 速度”列与详情保留组件参数量、硬件和原文测量口径；论文报告不是本地重测，缺失数据不猜测。不同 RTF 定义与采样预算不能直接排名。
- 星级分为 Zhanh 与 Other；Other 保留推荐人姓名，未评分留空，`*` 表示等待重读。解读颜色为 Zhanh 深红、Other／Felix 绿色、Hanyu 蓝色。
- 最后一栏“主要作者”列出第一作者单位，支持搜索、筛选、排序；点击单位可查看核验来源。
