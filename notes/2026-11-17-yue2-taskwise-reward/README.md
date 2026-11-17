# YuE2 歌手身份与任务 Reward 实验

**状态**：2026-11-17 实验提案，待 YuE2 音频校准与盲听。本文来自本次调研，示例与权重均为待验证的实验设计。

首要问题：对于同一独唱输入生成的多条歌，哪种 reward 能选出歌手更稳定的版本，同时保留正常换声区、力度变化和歌词覆盖？这个指标通过盲听后，再测它能否通过 GRPO 改善模型。任务扩展依次是纯音乐、明确轮唱、同时合唱。

| 文件 | 内容 |
| --- | --- |
| [REWARDS.md](REWARDS.md) | 每种 reward 的计算、适用任务、失败处理与组合公式 |
| [PROTOCOL.md](PROTOCOL.md) | 生成预算、人工听测、离线重排、训练对照和推进条件 |
| [prompt_seeds.jsonl](prompt_seeds.jsonl) | 32 个原创曲目种子，含语言、风格、歌词、数据划分和可用任务 |
| [annotations.csv](annotations.csv) | 听测标签模板。每个异常时段一条记录，支持曲目级判定和 A/B 配对 |
| [RUN_PROMPT.md](RUN_PROMPT.md) | 可直接复制到实验机聊天窗的执行指令 |

## 调研到实验的对应

- 长语音一致性使用曲内分窗声纹相似度。YuE2 的短暂 A→B→A 可能被全局平均稀释，因此本实验把均值与尾部分位、异常持续时间直接做对照。歌声特化 encoder 是首选评判器候选，由盲听校准。 [长语音评测](https://aclanthology.org/2026.findings-acl.112/) · [歌声身份编码器](https://github.com/SonyCSLParis/ssl-singer-identity)
- 歌手身份数、同时活跃的歌手数和声部层数有各自的含义。同人叠录可以有多个声部；轮唱和同时合唱需要明确段落计划，各角色跨段一致性与合唱覆盖分别验证。 [Singer Diarisation](https://www.isca-archive.org/interspeech_2024/munakata24_interspeech.html) · [Tutti](https://arxiv.org/html/2602.08233v1)
- 混合 GRPO 按完整输入分组计算 advantage，每项任务的有效组比例可能不同。仓库的旧入口采用 G=2，极小的评分差也能形成满幅优势。正式 CMI-Pref 配置包含 KL。运行前核对实际入口、采样和 KL 配置。 [MT-GRPO](https://proceedings.mlr.press/v306/ramesh26b.html) · [仓库 GRPO 记录](../../music-gen/lyrics2song/rl/grpo/README.md) · [正式 CMI-Pref 协议](../../music-gen/lyrics2song/rl/grpo/FORMAL-CMI-PREF.md)

产生真实运行后，按 [实验记录模板](../EXPERIMENT_RECORD.md) 保留配置、版本、预算和独立评估。
