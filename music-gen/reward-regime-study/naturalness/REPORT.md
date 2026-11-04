# 音乐生成不自然人工感的 reward 探索

**最推荐本域人工缺陷偏好 head，加 Audiobox PQ 质量约束；目前不能直接作为已验证的 RL 自然度目标。** 本轮跑通了25条音频的 Audiobox、PAM、SongEval 和信号诊断，但人工标签为0，没有训练新的自然度 head，也没有进行新 RL 训练。结果支持继续采标和影子评分，不支持“已经降低电音感”的结论。

播放器、原始分数与待评价位置在 [RL-Lab 自然度实验页](https://zhanh-he.github.io/RL-paper-reading/demos/#naturalness)。完整结果见 [audit.json](runs/2026-11-04/audit.json) 和 [逐样本表](runs/2026-11-04/sample-scores.csv)，19个候选的一手资料、权重与许可证见 [RESEARCH.md](RESEARCH.md)，复现入口见 [README.md](README.md)。

## 背景与目标

已完整阅读随后到位的 `MusicGenerationLLM_Discussion_Summary.docx`。背景关注连续表征驱动声学生成、非文本条件保持，以及实际生成路径与RL动作概率是否一致；本机推理入口已核到连续hidden→声学renderer连接，但缺完整renderer依赖、训练动作概率contract及可追溯目标音频。本文只公开泛化的方法问题，不转载内部设计。文档中的声学归因是待检验讨论，不能直接当实验事实。现有仓库的 [研究协议](../protocol.json)、[reward实现](../rewards.py) 与历史实验另行区分 AR token GRPO 和连续 Flow-GRPO；不同生成与解码路线不能默认共用同一动作概率或声学结论。

仓库已有 SP Richness-v1、SongBench Instrument/Arrangement 单目标，以及另行记录的 Mean7。SP 是频带活动、运动和调性门控，不是乐器计数；Instrument 衡量乐器真实感与合成保真，Arrangement 衡量配器。前三类目标减公共电平/近满幅惩罚，Mean7 的实现口径不同；KL 另属训练损失。SongEval 留作独立诊断。本轮沿用这些组织方式新增审计，没有替换既有目标。

“电音感”应拆成不想要的固定 buzz/哨音、金属或水声质感、颗粒与断裂、瞬态消失、vocoder/formant 不稳定、失真等。codec、上采样、生成误差和后处理都可能相关，但目前不能归因。[上采样伪影研究](https://arxiv.org/abs/2010.14356)支持部分算子产生 tonal artifacts 与频谱副本的机制；它不支持把正常谐波、合成器或明亮高频当缺陷。FM钟、镲片、失真吉他、chorus、Auto-Tune、pad 和 sidechain 都可能符合音乐意图。

仍需实际代码/运行记录确认：codec/renderer配置与原生采样率；RL更新与冻结模块；每个随机动作、其采样分布与old/new log-prob；连续条件随policy改变时是否遗漏梯度路径；已有生产reward尺度、非文本控制量与真实stems。单凭架构描述不能判定GRPO必然错误；但不能为没有实际采样的隐状态事后假设高斯动作分布，或拿无关token概率代替真实动作概率。

一手核实纠正了几项易误用判断：[Transfusion](https://arxiv.org/abs/2408.11039)是2024年的文本/图像混合目标，不是连续音频架构的通用同义词；[AudioLM](https://arxiv.org/abs/2209.03143)于2022年使用离散语义/声学token，非文本控制更早已有[DDSP](https://research.google/pubs/ddsp-differentiable-digital-signal-processing/)等工作。3–8kHz能量或固定谱峰不能证明某品牌AI签名；[FM原论文](https://arxiv.org/abs/2210.02747)回归的是条件向量场，其均值性质不推出高频必然变正弦。VAE相位引起梳状伪影也需同内容重建验证。依据Praat，[jitter](https://praat.org/manual/Voice_2__Jitter.html)是周期扰动，[shimmer](https://praat.org/manual/Voice_3__Shimmer.html)是逐周期幅度扰动，均非LPC共振峰平滑度；接近零不是通用机器检测规则。

[DDPO](https://arxiv.org/abs/2305.13301)使用随机去噪转移的policy gradient，不要求奖励可微；[Flow-GRPO](https://arxiv.org/abs/2505.05470)为此构造SDE。给定当前状态的确定性ODE转移没有可直接套入PPO的非退化高斯action log-prob，这不等于ODE最终样本没有密度。[MR-FlowDPO](https://arxiv.org/abs/2512.10264)提供音乐偏好目标的另一条路线，不能把回归surrogate称精确轨迹似然。可微reward穿过ODE是不同方案，需保留完整梯度链并控制数值误差，不能声称adjoint“无损”；当前离线scorer没有实现这条链。

## 样本与测量

共25条固定10秒片段：8条项目来源、2条录制参考、12条受控变换、3条构造信号。项目来源包括 Muse 原始/Mean7 step270、DiffRhythm2 原始/Mean7 step270、YuE2 原始/Richness step270 三组同输入配对，以及 ACE Studio Emma 合成歌声与 LaDA-Band 原始伴奏。前两组用既有 valid001，YuE2 用 valid000；预先固定来源，不按本轮分数选样。Emma 与伴奏有共同条件，不能当相同内容质量配对。

**这些样本不是背景目标连续表征系统的输出。** 本轮测得的是公开模型/已有项目材料上的评分响应，存在生成域与伪影类型差异；不能证明候选能识别目标系统缺陷。接入前必须追加目标系统同输入样本与重建对照，重新标注和校准。

录制参考为 Kevin MacLeod 的 [Vibe Ace](https://freemusicarchive.org/music/Kevin_MacLeod/Jazz_Sampler/Vibe_Ace) 和 Karissa Hobbs 的 [Let’s Go Fishin’](https://freemusicarchive.org/music/Karissa_Hobbs/Age_of_Flowers/09_Lets_Go_Fishin)，均按随源元数据标 CC-BY-3.0，分别取10秒和30秒起点。下载分发源本身是 OGG Vorbis，不能称录音母带。录制来源也不是“自然度已获人工确认”的标签。

Muse 原始与 Vibe Ace 各派生六条：−20dB增益、3kHz低通、RMS匹配静态压缩、RMS匹配硬削波、40Hz AM、与AM同RMS的纯增益控制。AM包络为0.10–1.00，比较时用后者控制总体电平。另构造静音、普通与明亮谐波合成器。已知处理不是主观缺陷真值；本轮没有虚构试听意见。

统一评分副本为48kHz双声道 float WAV，无外加响度归一化；模型再走各自前端。所有分数绑定评分副本SHA256。网页只发布48kHz、192kbps MP3试听副本，另存其hash；有损试听文件不与评分字节等同，也不因转成WAV就使上游源成为无损母带。

| 已运行方法 | 实际输入与分数含义 |
|---|---|
| Audiobox Aesthetics | mono→16k，单个10秒窗；PQ/PC/CE/CU训练标尺1–10。PQ是制作质量，PC是复杂度。 |
| PAM | 正确mono后→44.1k；[0,7]与[3,10]两窗分数均值，窗口重叠，不算两个独立样本。固定 clean/noisy-artifacts prompt 的softmax，0–1，非自然度概率。 |
| SongEval | 官方librosa mono→24k，MuQ隐藏层6；五头为 Coherence、Musicality、Memorability、Clarity、Naturalness，输出(1,5)。Naturalness语义是人声呼吸/分句，Clarity是歌曲结构，不是全混音伪影检测。 |

定义分别依据 [Audiobox论文与代码](https://github.com/facebookresearch/audiobox-aesthetics)、[PAM论文](https://arxiv.org/abs/2402.00282)和 [SongEval实际API](https://github.com/ASLP-lab/SongEval/blob/main/eval.py)。不同采样率、窗口和任务域保留不变，不能称统一声学带宽。十秒也不足以评定完整歌曲结构和记忆性。

## 实际观察与失败例

下表为处理后减比较对象的自动分数变化；AM只减等RMS纯增益控制，其余减原片。分数越高仅表示该评估器对应维度越高。

| 来源与处理 | ΔPQ | ΔPAM | ΔSongEval Musicality |
|---|---:|---:|---:|
| Muse 40Hz AM／等RMS控制 | −1.724 | −0.078 | −0.239 |
| Vibe Ace 40Hz AM／等RMS控制 | −1.780 | −0.183 | −0.308 |
| Muse 硬削波 | −0.723 | −0.080 | +0.097 |
| Vibe Ace 硬削波 | −4.560 | −0.609 | −0.515 |
| Muse −20dB | +0.006 | −0.145 | −0.141 |
| Vibe Ace −20dB | +0.166 | −0.003 | −0.318 |
| Muse 3kHz低通 | −0.262 | −0.333 | +0.085 |
| Vibe Ace 压缩 | −1.669 | −0.020 | −0.174 |

PQ对这两条AM和硬削波有反应；PAM对同一处理的响应明显依赖内容。Muse削波后SongEval Musicality反而增加，说明它不能替代局部技术缺陷判断。低通和压缩同样不能预先强判为更差：某些风格需要这些效果，仍须听者确认。

静音的PQ为 **6.735**、PAM **0.354**、SongEval均值 **1.064**。有数值不等于有效音乐：整条静音须先过任务有效性检查，正常休止不能照此惩罚。普通→明亮合成器的PQ为7.185→7.180，PAM为0.766→0.694；这提示合法音色保护的重要性，但没有人工标签，不能报告误伤率。该构造还同时改变谐波结构和RMS约1.18dB，不能隔离为纯明亮度因果。

另试了自拟“natural without unwanted digital artifacts／unnatural metallic buzzing”prompt对；**16/25分数大于0.99，静音也得0.942**。该探针饱和且缺少有用语义区分，本轮淘汰为直接reward，原分数只留作失败记录。初版AM经RMS放大后有越界峰值，已改为受限包络加等RMS控制，相关评分全部重跑；本报告不使用旧结果。

三组真实生成配对进一步显示proxy分歧：

| 既有后期checkpoint减原始起点 | ΔPQ | ΔPAM | ΔSongEval Musicality |
|---|---:|---:|---:|
| Muse Mean7 step270 | −0.244 | +0.048 | +0.148 |
| DiffRhythm2 Mean7 step270 | −0.571 | −0.013 | +0.026 |
| YuE2 Richness step270 | +0.134 | +0.025 | −0.630 |

这是每组一个十秒片段的重新评分，不是完整验证集结果；不能用它宣称这些历史RL方案改善或损害听感，更不能归因为本轮自然度reward。人工标签当前为 **0**，Spearman和pairwise accuracy均为空，没有估计人类一致性。自动指标之间相关也不能填补这一缺口。

## 信号诊断能解释什么

[signal_features.py](signal_features.py)只输出描述性特征，没有拼成“电音总分”。

| 特征 | 可能相关之处与正常声音误伤 |
|---|---|
| 近满幅比例、越界比例、peak/RMS | 查数字电平问题；减增益可避开满幅阈值，失真吉他或限幅可合法。满幅比例不等于通用失真测量。 |
| flatness、8k以上能量、95% rolloff | 描述噪声性与带宽；镲片、气声、明亮synth正常，低通容易刷低高频值。 |
| 3–8kHz能量与固定谱峰持久性 | 已补跑，分别量化能量占比及局部突出谱峰跨帧出现比例；稳定谐波、齿音和合成器同样可高，不能标记机器来源或缺陷真值。 |
| spectral flux | 此处是L1归一幅谱的相邻帧L2变化，非标准正向mel onset flux；pad变化少、鼓变化多都可正确。 |
| 包络20–100Hz调制与周期性 | 探索buzz/调制；beat、tremolo和sidechain也会升高。 |
| crest与动态分位差 | 探索过度压缩；动态差是40ms活跃RMS的p95−p10，非EBU LRA；正常电子乐可强压缩。 |

立体声折叠能量单独记录，mono抵消时谱特征为空，不制造NaN或假分。本轮**未实现或验证**多声部谐波/F0异常、谱包络/formant、固定谱峰的伪影归因、子带金属感、瞬态损失、true peak和感知空间质量；不会把缺失项说成已测。[谱平坦度官方定义](https://librosa.org/doc/0.11.0/generated/librosa.feature.spectral_flatness.html)也只描述noise-like与tone-like，不证明自然度。

新增的3–8kHz固定峰诊断，用2048点STFT，在可测带内能量帧中查找超过13频点局部中值10dB的谱峰，取固定频率上的最大出现比例；它没有跟踪F0，不能分离持续乐音与伪影。明亮合成器为0.801，Muse起点为0.243，Vibe Ace为0.276。这个音色控制足以提醒：惩罚稳定高频峰会伤害本应保留的谐波音色，不能按这些数字宣布谁更自然。低带能量或mono抵消时返回空值而非“无缺陷”。

## 推荐与接入条件

1. **本域偏好head作为主路线，尚未训练。** 用冻结embedding的L2归一特征，`u(x)=wᵀf(x)`；以“较少不想要人工感”为正向，优化 `−log σ(u(A)−u(B))+L2正则`。[preference_head.py](preference_head.py)提供线性BT实现，强制独立song/prompt分组注册与端点检查，避免同源切片跨集合。小head思路可参考 [TuneJury](https://arxiv.org/abs/2606.17006)，不能把其整体偏好能力当本域缺陷验证。
2. **Audiobox PQ作质量约束并保留独立音乐性检查。** 当前先影子评分。它对部分处理敏感，但静音高分、16k mono带宽和空间盲区明确存在，不把PQ改名自然度。[Audiobox原论文](https://arxiv.org/abs/2502.05139)支持广谱质量用途，未提供本域人工感保证。
3. **PAM保留探索性对照或head辅助特征。** 暂不单独驱动RL；优先检验音量、合法合成器、过度滤波取巧。自拟prompt变体已淘汰。语音MOS、AI来源检测和FAD不作为直接替代，原因见[候选比较](RESEARCH.md)。

[reward_adapter.py](reward_adapter.py)是可调用示例，尚未接入trainer。对existing、head logit、PQ分别冻结校准集p05/p95，使用 `S(v)=clip(2(v−p05)/(p95−p05)−1,−1,1)`，得到 `b,n,q`。当前示例为：

```text
R = clip(b + λ min(n,q) − 0.1 max(0,−0.5−q) − optional_clip_penalty, −1,1)
c = near_full_scale_fraction
optional_clip_penalty = 0（默认已有shared guard）
                     或 0.1 min(1,c/0.01)（显式指定本层为唯一guard owner）
```

这是显式改变旧reward尺度的候选组合，阈值/惩罚尚未校准；`shared_guard_already_applied=True`默认不二次扣近满幅惩罚，接入时必须核对所选旧目标并明确一个guard owner。缺head或校准抛错，失效/任务覆盖失败拒绝；同时记录未裁剪值与饱和。p05/p95不能逐batch重算，head logit和PAM均不能无标签称概率。

完整曲拟用10秒窗、5秒hop，示例聚合 `0.7 mean+0.3 Q20`，保留差片段；当前只测单个十秒，尾段时长权重尚未实现。**现在λ=0仅表示不改变既有训练**，不是已有head可运行；验收后才扫0.02/0.05/0.1。需同时约束整体质量、独立音乐性、节奏/动态、条件保持与多样性，检查沉默、降音量、低通和配器简化等捷径。[MusicRL](https://arxiv.org/abs/2402.04229)及[多维rubric研究](https://arxiv.org/abs/2610.03589)支持多目标与独立评价的必要性，不能代替本轮对照训练。

## 成本、复现与下一步

| 本轮环境 | 10秒片段中位耗时 | 初始化 | 进程峰值RSS |
|---|---:|---:|---:|
| M4 CPU四线程／Audiobox | 0.196s | 1.13s | 1.35GB |
| M4 CPU四线程／PAM | 0.066s | 3.16s | 1.80GB |
| Linux Ryzen 9 7900 CPU四线程／SongEval | 1.050s | 1.88s | 3.62GB |

M4为24GiB统一内存，未用GPU/MPS；前两项计时排除文件解码，含模型前处理/前向等，PAM含两重叠窗。SongEval含librosa解码重采样，硬件不同，**不能组成速度排行榜**；初始化不含下载与Python导入，缓存也可能已热。Signal诊断含解码中位0.088s（含新增3–8kHz谱峰诊断）。均为25条顺序batch-one测量，不能外推训练吞吐或GPU显存。依赖、hash、失败原因与命令见README和运行记录；HF下载停滞后使用官方S3，PAM依赖版本调整后核对权重键覆盖，SongEval用现有Linux缓存完成独立评分。

本轮只公开MP3及脱敏清单；录制参考保留署名与CC-BY链接，构造信号CC0。模型源码、权重、数据许可分别核对：Audiobox主体CC-BY4；PAM源码MIT但MS-CLAP权重MS-PL；SongEval CC-BY-NC-SA4、MuQ权重CC-BY-NC4，不能据此承诺商业生产部署。其他候选限制见RESEARCH。

下一步先在页面真实听评，随机A/B、允许tie/不知道，分别填人工感、音质、音乐性，隐藏模型与自动分数，重复10%测一致性。head探索建议40个独立prompt/源歌曲组、每组3对、两位听者，约120个独立pair和240条判断，按组预固定24/8/8训练/校准/测试。该规模仍探索性；重复投票不增加独立pair数，人工变换不冒充真实生成缺陷。报告分组bootstrap、风格误伤与质量取舍，过验收再做有baseline、独立听测的短RL对照。并补充同内容codec重建及同latent跨renderer对照，才能判断reward应作用生成器、解码器还是数据选择。
