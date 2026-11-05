# 调制与 roughness：108 个真实输出的聚合混杂审计

2026-11-05 · 人工标签 **0** · 新增音频、模拟、训练 **0**

**结论：当前全带调制 M 和 roughness 适合定位试听，尚不能直接作为“少一点电音感”的 reward。全带均值可能掩盖频段方向相反的变化，roughness 均值也可能由少数时间窗主导。**

[打开 RL-Lab 试听与评价位置](https://zhanh-he.github.io/RL-paper-reading/demos/#naturalness)。先判断变化是否属于不想要的人工伪影，还是正常合成器、鼓击或失真音色；来源和指标方向都不是主观标签。

## 全带 M 会掩盖相反的频段变化

按当前实现，将每个子带/时间点用 `envelope_mean_relative² × (envelope_mean_relative > 0.001)` 聚合。以下频段指 gammatone **中心频率分组**；权重是平方包络均值，不是校准声能或响度。

| 真实输出集合 | 片段数 | <1.5 kHz 对全带 M 的贡献占比中位数 | 3–8 kHz 贡献占比中位数 | 低频组贡献 >50% |
|---|---:|---:|---:|---:|
| Suno V5 | 48 | 81.1% | 8.5% | 48/48 |
| DiffRhythm2 | 48 | 64.9% | 14.2% | 38/48 |
| SongBloom 社区生成 | 8 | 76.9% | 8.9% | 8/8 |
| Muse / 公开 MuCodec 路由 | 4 | 80.7% | 7.3% | 4/4 |

108 条中没有一条由 3–8 kHz 组贡献超过一半。这说明当前聚合更受低频组影响，**没有证明**目标伪影位于任何特定频段。

公开 Muse/MuCodec 的实际底层 guidance 对照固定了 codes 和 decoder noise：

| 实际 decoder guidance | 全带 M | <1.5 kHz 组内 M | 低频组聚合权重 | 3–8 kHz 组内 M | DW70 roughness 均值 |
|---|---:|---:|---:|---:|---:|
| 1.0 | 0.25838 | 0.23544 | 89.8% | 0.45663 | 0.09943 |
| 1.5 | 0.23890 | 0.20987 | 90.5% | 0.51088 | 0.18578 |
| 2.0 | 0.23035 | 0.19891 | 91.1% | 0.54829 | 0.22877 |

1→2 时，全带 M 下降约 **10.8%**，3–8 kHz 组内 M 却增加约 **20.1%**。这是数值聚合中的相反方向，不代表听感已变好或变坏。若目标后来被确认主要是高频人工感，`−全带 M` 有偏离目标的可能。也不能直接换成 `−高频 M`：正常合成器、镲片等声音的误伤尚未排除。

## roughness 均值不能说明它持续多久

计算每段最高 10/99 个 DW70 窗口占 roughness 窗口和的比例：

| 输出集合 | 最高 10 个窗贡献占比中位数 | 最高 10 个窗贡献 >50% |
|---|---:|---:|
| Suno V5 | 50.5% | 24/48 |
| DiffRhythm2 | 57.9% | 36/48 |
| SongBloom | 45.8% | 3/8 |
| Muse / MuCodec | 66.1% | 4/4 |

这是 200 ms、100 ms hop 的重叠分析窗口，不是精确事件时长。峰值位置适合引导试听；目前不能判断它们是真伪影，还是正常鼓击、和弦干涉或失真。JSON 的 `frames_gt1asper` 仅用于描述，没有把 1 asper 设为经过验证的缺陷阈值。

## 24 Hz 峰暂不能用来诊断伪影

`fast_modulation_peak_hz` 报 24 Hz 的数量为 Suno 38/48、DiffRhythm2 45/48、SongBloom 6/8、Muse 4/4，共 **93/108**。当前搜索从 20 Hz 起，使用 5-bin 平滑及边界填充；24 Hz 可能受下界和谱形影响。本审计没有完整调制功率谱，不能确认成因，更不能称其为跨模型共有的 24 Hz 伪影。

## 下一步与证据边界

保留原音频、局部时间点、分频段 M、聚合权重与 roughness 时间分布，先完成既有同内容对照的听评。同时评价不想要的人工感、整体音质和音乐性，允许“正常音色”与“无法区分”。通过听评后才讨论校准与 reward 权重。

- 108 个片段包含同曲窗口和同 prompt/seed 参数变体，不能当作 108 次独立重复；没有置信区间或检测准确率。
- gammatone 频段有重叠；组内 M 是探索性描述量，其升降不是人工伪影判决。
- 没有主观标签，没有正常合成器误伤率，没有 RL/GRPO 提升结论。
- 公开实现的参数及机制结果不能外推到其他实现或内部系统。

## 数据与复现

只读取已公开的分数文件；`numpy` 为唯一非标准库依赖。在仓库 `music-gen/reward-regime-study/naturalness/` 中运行下列命令（将本脚本置于该目录）：

```sh
python descriptor_confound_audit.py \
  --suno-auditory runs/2026-11-04-suno-real/features.json \
  --suno-roughness runs/2026-11-04-suno-real/features.json \
  --controls-auditory runs/2026-11-05-controls/raw-scores.json \
  --controls-roughness runs/2026-11-05-controls/raw-scores.json \
  --output descriptor_confound_audit.json
```

- [脚本](descriptor_confound_audit.py) · [完整结果及输入 SHA256](descriptor_confound_audit.json)
- [Suno 原始描述量](https://github.com/zhanh-he/RL-paper-reading/blob/main/music-gen/reward-regime-study/naturalness/runs/2026-11-04-suno-real/features.json) · [真实生成对照原始描述量](https://github.com/zhanh-he/RL-paper-reading/blob/main/music-gen/reward-regime-study/naturalness/runs/2026-11-05-controls/raw-scores.json)
- [Suno 样本来源与方法](https://github.com/zhanh-he/RL-paper-reading/blob/main/music-gen/reward-regime-study/naturalness/SUNO_REAL_REPORT.md) · [同输入、guidance 对照与局限](https://github.com/zhanh-he/RL-paper-reading/blob/main/music-gen/reward-regime-study/naturalness/PROMPT_CONTROL_REPORT.md)

108 条 auditory / roughness 的音频 SHA256 均逐条匹配，全带 M 重构最大绝对误差 `6.71e-8`。分开保存的 worker 结果与现有仓库合并文件均复现了完全一致的所有数值结果。
