# “少一点电音感”：外部证据与下一步

2026-11-05

**目前最可靠的建议：把这些 psychoacoustic 指标用来定位和解释声音差异，暂不直接优化它们。新增 artifact reward 权重保持为 0。** 我们已找到可以在固定内容下改变声音的控制位置，但还没有经过听感验证的“人工伪影分数”。内部 renderer/codec 未参与本轮实验，外部结果不能直接外推其参数、机制或效果。

你可以等本轮工作全部完成后再集中听；无需现在同步操作。届时先听下面四组真实 Suno，再听两类生成对照，留下“这就是我要压的声音”和“这是正常电子音色”各一个明确时间锚点。本轮实验及整理已完成，听评留待你集中判断。

## 已经确认的事实

- 样本包括发布者标为 Suno V5 的 16 首歌曲、48 个固定片段；另有 32 个公开模型生成分析输出、60 个固定评分窗口。**人工标签仍为 0**，没有伪影检测准确率、reward 训练或 RL 改善结论。不同窗口和同条件多档输出也不是独立歌曲。
- 同一组 750 个 Muse 音频 token、相同 decoder 随机噪声下，公开 MuCodec 的实际 guidance 从 1 到 2：全带调制 M 从 **0.25838 降至 0.23035**，roughness 从 **0.09943 升至 0.22877**，质量代理 PQ 从 **7.621 升至 7.978**。说明控制确实改变了描述子，同时说明“roughness 越低越好”尚不能成立。
- 新的分解审计发现：上述全带 M 下降约 **10.8%** 时，中心频率 3–8 kHz 子带的组内 M 却上升约 **20.1%**。原 48 个 Suno 片段中，中心频率低于 1.5 kHz 的组对全带 M 贡献占比中位数为 **81.1%**。全带均值会掩盖频段相反变化；这也不代表应直接奖励低高频 M，正常镲片、失真和合成器尚未排除。
- roughness 也需要看时间：48 个 Suno 片段中有 **24 个**，最高 10 个重叠分析窗贡献了整段 roughness 窗口和的一半以上。峰值有助于选听，可能对应正常鼓击、和弦干涉或缺陷，当前没有人的判定。常见的 **24 Hz** 调制峰还可能受搜索下界、平滑和谱形影响，不能解释为共同的“24 Hz 伪影”。
- 在本次**公开归档版本**的 MuCodec 中，外层 guidance 传 1 或 2，实际下层都收到 1.5，得到逐位相同的解码 PCM；完成实例级参数路由后才得到有效对照。这是公共版本的工程事实，**不代表内部实现存在相同问题**。内部值得迁移的是“核查实际生效参数”的方法。

M 是自定义子带调制描述子；DW roughness 采用假定的 70 dB RMS SPL，不是实际播放声压校准；孤立共振是自定义窄峰/持续性代理，均不是缺陷概率。分解审计核对了 108 条既有片段的输入指纹，全带 M 重构最大绝对误差为 6.71×10⁻⁸；这是统计复核，没有新增听感标签或生成样本。

## 你随后只需完成的试听

先分别判断“不想要的人工感、整体音质、音乐性”，再展开自动指标；允许“相当/无法判断”。正常合成器、明亮高频、失真吉他都不能因为电子音色而自动扣分。

| 精选 | 原曲 A / B 的时间 | 重点 |
|---|---|---|
| [Suno 01](https://zhanh-he.github.io/RL-paper-reading/demos/?nat-pair=suno-listen-1#naturalness) | cn000001_0：02:32.832–02:42.832 / cn000001_1：02:25.984–02:35.984 | 共振分相差约 84 倍、roughness 接近：听到的是讨厌窄带声音，还是乐器/音高差异？ |
| [Suno 02](https://zhanh-he.github.io/RL-paper-reading/demos/?nat-pair=suno-listen-2#naturalness) | en034001_0：00:50.968–01:00.968 / en034001_1：00:51.976–01:01.976 | roughness 相差 3.36 倍：是否只是应保留的失真吉他？ |
| [Suno 03](https://zhanh-he.github.io/RL-paper-reading/demos/?nat-pair=suno-listen-3#naturalness) | cn008001_0：00:31.960–00:41.960 / cn008001_1：00:42.800–00:52.800 | 正常电子/合成器条件：调制差异是否来自节奏与合法音色？ |
| [Suno 04](https://zhanh-he.github.io/RL-paper-reading/demos/?nat-pair=suno-listen-4#naturalness) | cn008001_0：01:19.900–01:29.900 / cn008001_1：01:47.000–01:57.000 | 三项指标接近：若人工感明显不同，就是当前特征缺失的线索。 |

四组共约 80 秒原片，可反复 A/B。这些来自同一来源家族，尚未证实原始输入一致，乐句也不严格对齐，不能当成单因素因果实验。

然后听两类生成对照：

1. [Muse：同一音频 token、相同噪声，实际 renderer guidance 1 / 2](https://zhanh-he.github.io/RL-paper-reading/demos/?nat-corpus=controls&nat-pair=muse-renderer-low-high#naturalness)。重点看声音差异是否就是目标缺陷，而不是质量、亮度或配器变化。
2. [同一 Suno 音色参考的 DiffRhythm2 / SongBloom](https://zhanh-he.github.io/RL-paper-reading/demos/?nat-corpus=controls&nat-pair=model-synth-audio-ref#naturalness)。分别评“Suno 音色相似度”和“不想要的人工感”，另看所需配器是否保留；模型来源不作坏标签。

网页评分保存在本机，可导出 JSON 留存，稍后异步分析。

本轮又用 RTX 5090 完成 **8 次 DiffRhythm2 真实生成**（含 2 次条件相同的检查），增加 16 个固定窗口。采用两条件引导 `v₊ + β(v₊ − v₋)`，两分支保留同一歌词和配器；两个检查的 PCM 与中间表示均逐位相同。原声条件 β0→1 的 DW70 为 0.0412→0.0367，合成器条件却为 0.0925→0.1147，全带调制在两者中都下降：再次说明单一指标方向不能当作成功标签。**这轮验证了可运行的控制方法，尚未证明听感改善。** [新试听](https://zhanh-he.github.io/RL-paper-reading/demos/?nat-corpus=contrast&nat-pair=contrast-synth-contrast05#naturalness) · [实测与复现](https://github.com/zhanh-he/RL-paper-reading/blob/main/music-gen/reward-regime-study/naturalness/CONTRAST_FOLLOWUP_REPORT.md)。只有一套歌词、两种风格、一个 seed；未证明所需配器保留。

## 给内部同事的三步建议

1. **先核查参数与重复性。** 保存在内部的一份上游条件与 decoder 随机状态，重复基线，再改变一项实际生效参数。记录请求值、下层值、采样设置、PCM 差异与耗时；参数未生效或重复波动未厘清就停止解释听感效果。
2. **固定内容做有限对照，保护正常音色。** 先用 2 个条件跑通；再考虑 4 个条件（含正常合成器及人声/原声）×3 档有效设置×2 个固定噪声的探索网格。内部决定参数范围，不照搬公开模型档位。若指标改善伴随合成器丢失、低通、低音量、节奏弱化或质量下降，该方案不计成功。
3. **有真实听感锚点后再验证 reward。** 按整曲/输入家族分组留出，先做只评分、候选排序，再考虑受控 RL/GRPO。推荐方向是用局部子带调制、roughness 时间峰、共振与已有音频表征，校准小型缺陷偏好模型；这是待验证方案。没有独立听评支持前，新增 artifact 项权重维持 0。

上述规模只是起步建议，并非统计功效保证；接受幅度、质量及配器容忍度需在复验前约定。内部模型、代码、条件和音频均可留在内部，外部提供可复现方法与公共证据。

资料：[实验与复现](https://github.com/zhanh-he/RL-paper-reading/blob/main/music-gen/reward-regime-study/naturalness/PROMPT_CONTROL_REPORT.md) · [Suno 精选说明](https://github.com/zhanh-he/RL-paper-reading/blob/main/music-gen/reward-regime-study/naturalness/LISTENING_REVIEW_2026-11-05.md) · [公开 MuCodec 参数调用](https://github.com/tencent-ailab/MuCodec/blob/128f91be6c1457fd4b8822042ec5994d6d58a70b/generate.py#L170)。
