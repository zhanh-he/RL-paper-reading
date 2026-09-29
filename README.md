# Music Post-Training Lab

[Literature catalog](https://zhanh-he.github.io/RL-paper-reading/) · [Live experiment demo](https://zhanh-he.github.io/RL-paper-reading/demos/) · [Today's presentation status](notes/2026-09-30-presentation-status.md) · [Request an update](upd_request/) · [Editing guide](platform/docs/EDITING.md)

This repository keeps the shared RL literature catalog and measured music post-training pilots. The [experiment demo](https://zhanh-he.github.io/RL-paper-reading/demos/) has audio and MIDI replays with versioned receipts. It labels local measurements, reward diagnostics, baseline inference and unfinished GRPO work separately.

## Experiment Map

| Area | Task | Current state |
| --- | --- | --- |
| [Music transcription](music-trans/README.md) | [Multi-instrument](music-trans/multi-inst/README.md) | [MuScriptor hosted baseline](music-trans/multi-inst/models/muscriptor/README.md) on the public synthetic SATB example; one piano track, no local training access yet |
| [Music transcription](music-trans/README.md) | [Choral singing](music-trans/choral-singing/README.md) | [Frame-head GRPO and full-song 50 ms note F1](music-trans/choral-singing/rl/grpo/runs/2026-09-29-frame-head/README.md), 30-song held-out test; [public input/MIDI/audio replay](music-trans/choral-singing/rl/grpo/runs/2026-09-30-public-synthetic/receipt.json) illustrates unchanged event decoding; [four-pair note reward audit](music-trans/choral-singing/rewards/audits/2026-09-29/README.md) |
| [Music generation](music-gen/README.md) | [Fixed vocal to accompaniment](music-gen/vocal2accomp/README.md) | [ACE-Step 1.5 completion](music-gen/vocal2accomp/models/ace-step-1.5/README.md) and [AnyAccomp isolated accompaniment](music-gen/vocal2accomp/models/anyaccomp/README.md) on the same synthetic guide; coverage/beat-v2 measured on AnyAccomp, no GRPO yet |
| [Music generation](music-gen/README.md) | [Lyrics to song](music-gen/lyrics2song/README.md) | [YuE2 SongEval-GRPO](music-gen/lyrics2song/rl/grpo/yue2_songeval_longrun.py) measured at 0/1/5/50 updates, 100-update run in progress; [Muse SongEval pilot](music-gen/lyrics2song/rl/grpo/runs/2026-09-29-muse/README.md) with audio pair; [MuseCritic online-GRPO one-step run](music-gen/lyrics2song/rl/grpo/runs/2026-09-30-musecritic-smoke/receipt.json) and [held-out replay](music-gen/lyrics2song/rl/grpo/runs/2026-09-30-musecritic-heldout/receipt.json) completed on Gadi |

Each task owns `models/`, `rewards/` and `rl/{dpo,grpo}/`. A model folder holds an adapter/config and provenance, not weights. Simple reward implementations belong in one Python file each; `combine.py` at the reward root composes them after their scales and failure modes have been checked. A run gets its own folder under the appropriate RL method, with a reproducible record using the [experiment template](notes/EXPERIMENT_RECORD.md). [Model and data selection](notes/2026-09-29-model-data-reward-selection.md) records gates, licenses and recommended next experiments.

## Literature And Updates

1. 在 [interactive catalog](https://zhanh-he.github.io/RL-paper-reading/) 搜索、筛选和排序文献。
2. 修改请求写进 [`upd_request/`](upd_request/) 里自己的 TXT：[Zhanh](upd_request/Update_request_zhanh.txt)、[Felix](upd_request/Update_request_felix.txt)、[Hanyu](upd_request/Update_request_hanyu.txt)。完成的请求归档在 [`done_requests/`](upd_request/done_requests/)；这次搬目录没有处理待办内容。新实验方法直接归入所属任务的 `rl/`，不再维护第二套请求目录。
3. 团队可以直接编辑 [`platform/data/literature.csv`](platform/data/literature.csv)。`platform/site/data/literature.json` 是生成文件，不要手改；具体字段见 [editing guide](platform/docs/EDITING.md)。

The repository is public. Do not commit private Obsidian notes, restricted datasets, model checkpoints, credentials or audio without redistribution rights. Published paper results, local measurements and proposed experiments must be labeled separately.

## Labels
- Domain：`MusicGen / MusicEval / MIR / AudioGen / Speech / SpeechEnhance / AudioLLM / LLM / CV / MachineLearning / SourceSep / Multimodal / Other`
- Workstream：`Reward / RL / Reward-n-RL / Other`
- Zhanh 与 Felix 的星级分别由本人维护；`*` 表示评分存疑、等待重读。三人的解读颜色为 Zhanh 深红、Felix 绿色、Hanyu 蓝色。
