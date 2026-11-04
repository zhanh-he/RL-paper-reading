# 音乐生成不自然人工感 reward 候选核实

核实日期：2026-11-04。只使用原始论文、作者仓库、官方权重卡和库文档。已全文阅读随后到位的背景 Word；下文只给出泛化的方法核实，不转载内部设计。本文是研究筛选，不包含本机实验结果；实测与目标系统域差异见 [REPORT.md](REPORT.md) 和运行记录。背景中的归因与检测阈值不是已验证事实。

## 结论与边界

最值得落地的路径是「本域缺陷偏好 head + 广谱质量约束 + 音乐性约束」。本轮可立即运行的 Audiobox Production Quality（PQ）与 PAM 用于筛样、对照和发现失败，不应命名为已经验证的“电音感检测器”。DSP 指标保留为可解释诊断，不能未经标签验证直接相加成自然度。

不自然人工感至少拆成：固定频率 buzz/哨音与镜像频谱、金属/水声/相位涂抹、颗粒/断裂、过度平滑与瞬态消失、vocoder/formant 不稳定、混音失真/削波。上述听感可能来自 codec/decoder、上采样、生成分布、数据或后处理；没有同输入的中间层与重建对照，不能归因给某个模块。[上采样伪影原始研究](https://arxiv.org/abs/2010.14356)解释了算子可引入 tonal/filtering artifacts 与 spectral replicas；[BigVGAN](https://arxiv.org/abs/2206.04658)说明 anti-aliased periodic inductive bias 的生成价值。这些不是“所有谐波都是缺陷”的证据。

真实电子乐器、失真吉他、FM 钟、镲片、vibrato、chorus、Auto-Tune、lo-fi、gated synth、sidechain pumping 都可能是正确内容。训练目标必须是“与 prompt/编曲意图不符且听者认为不自然的缺陷”，而不是“像合成音”。

## 背景讨论的一手核实与修正

背景文档帮助明确连续表征、声学renderer及动作概率一致性问题，但不足以证明某种声学机制。推理入口连接已作本机只读核查，完整renderer依赖和训练概率contract仍缺；本轮25条为公开模型/既有项目代理样本，并非该目标链产物，结果不能直接外推。

### 术语与时间线

[Transfusion 原论文](https://arxiv.org/abs/2408.11039)于2024年提出在同一transformer上结合离散文本next-token目标与连续图像diffusion目标。它支持“混合离散与连续生成”的对标，不意味着任意hidden-conditioned音频renderer就是Transfusion复现，亦不证明消除量化瓶颈必然改善本域声学缺陷。

[AudioLM](https://arxiv.org/abs/2209.03143)首次预印本为2022年9月，采用离散语义和声学token；不能据此称它开创全部非语义或连续条件控制。[DDSP（2020）](https://research.google/pubs/ddsp-differentiable-digital-signal-processing/)已有pitch/loudness分离操控，足以反驳“底层声学控制始于AudioLM”的广义时间线。音乐旋律条件和连续latent提示也不是相同技术概念。

### RL先核对真实动作再选算法

连续表征进入renderer时，离散token概率是否仍是有效policy score，取决于真实随机变量及因果路径。若token仍被实际采样并控制生成，token REINFORCE不因输出是连续音频就自动失效；若参数又直接改变给定token下的hidden/renderer条件，则单一score-function项可能漏掉条件路径。若token根本不参与该rollout，拿它的概率做ratio没有相应动作依据。这里是应由代码与梯度验收核实的风险，不是已经证明训练错误。

| 路径 | 一手依据与必要条件 |
|---|---|
| DDPO／随机去噪policy gradient | [论文](https://arxiv.org/abs/2305.13301)、[作者实现](https://github.com/kvablack/ddpo-pytorch)：把去噪当MDP，对真实随机转移算log-prob；黑箱reward即可，不是将奖励本身沿ODE反传。 |
| Flow-GRPO | [论文](https://arxiv.org/abs/2505.05470)、[SDE log-prob实现](https://github.com/yifan123/flow_grpo/blob/main/flow_grpo/diffusers_patch/sd3_sde_with_logprob.py)：构造随机SDE转移，存均值/方差、时间表与样本，重算同一分布。确定性ODE下一状态条件于当前状态是退化转移，不能直接套普通高斯ratio；ODE端点可有CNF密度是另一问题。 |
| Diffusion-DPO／音乐Flow-DPO | [Diffusion-DPO](https://arxiv.org/abs/2311.12908)、[MR-FlowDPO](https://arxiv.org/abs/2512.10264)及[官方代码](https://github.com/lonzi/mrflow_dpo)：通过配对数据与推导的去噪/速度回归surrogate做偏好优化。surrogate不是声称测得精确波形似然；需核对参数化、路径、参考模型和时间权重。 |
| 连续动作或可微renderer路线 | 连续动作必须真的从明确参数化分布采样，不能对确定性hidden事后补一个高斯log-prob。可微reward路线需要整条renderer/reward计算图；[torchdiffeq](https://github.com/rtqichen/torchdiffeq)支持adjoint，但求解容差、稳定性与离散化误差仍存在，不能称“无损梯度”。当前离线scorer未实现这一路径。 |

接入前记录实际采样变量、支持集、old/current/reference概率、detached与可微条件、随机seed、CFG/solver、有效长度与KL单位。先验收零更新ratio、非零参数扰动后的敏感性、冻结模块和音频hash；仅有奖励接口不等于通过这些验收。

### 声学归因与检测规则

- **3–8kHz不是通用AI签名。** 未找到支持跨Suno版本、风格和音源的固定频段充分判据的一手证据。该区域也有正常谐波、齿音和打击乐；来源检测的标签是生成来源，不是人耳不喜欢的人工感。[SONICS](https://arxiv.org/abs/2408.14080)的任务定义不能转换成自然度真值。本轮新增频段占比和固定谱峰持久性仅为诊断。
- **FM的条件均值不推出正弦坍塌。** [FM原论文](https://arxiv.org/abs/2210.02747)回归概率路径的向量场，推断从噪声积分生成分布。由此不能把监督回归的平均效应等同于输出波形必然变正弦；零均值随机残差的均值更不必是周期波。后者是对损失对象的数学区分；噪声减少、过度平滑或过强调性是否发生仍要测。
- **相位问题可能但未归因。** VAE压缩、上采样、overlap/chunking及非线性都可能改变时间频率细节；没有同内容encode–decode、固定latent与边界对照，不能断言“振幅相位未解耦导致梳状滤波”。[上采样伪影原始研究](https://arxiv.org/abs/2010.14356)提供的是一种具体机制，不能当所有金属感的解释。
- **jitter/shimmer不是LPC共振峰抖动。** [Praat jitter](https://praat.org/manual/Voice_2__Jitter.html)定义周期时长扰动，[shimmer](https://praat.org/manual/Voice_3__Shimmer.html)定义逐周期幅度扰动，通常用于持续元音；[Burg/LPC formant](https://praat.org/manual/Sound__To_Formant__burg____.html)是另一个谱包络估计。接近零不能直接判机器，低抖动的稳定音、真实乐器和合法合成器都是反例；混合音乐的周期跟踪尤其不满足前提。本轮未用其作reward。

非文本条件保持应另设约束：可用真实voiced声源的F0误差、容差内onset/beat匹配、动态包络误差等，按具体条件定义。相关系数会忽略整体增益，单一embedding会混入内容/音色，FAD是分布比较；不能把任何一项直接叫通用conditioning adherence。自然度优化不能通过抹掉原控制信号得高分。

## 候选比较

表内“可运行”表示公开实现存在且技术上支持本地，不等于本轮已经跑通。许可证是来源声明与依赖核对，不代替使用方判断。

| 候选 | 实际测量、输入输出与参考要求 | 开放性/本机路径 | 决策与误伤 |
|---|---|---|---|
| **Audiobox Aesthetics** | 无参考；语音/音乐/音效；PQ/PC/CE/CU 四轴，训练标尺 1–10。PQ 是技术质量，PC 是内容复杂度，不能把 PC 越高解释成越自然。官方自动 mono→16k，10s 非重叠窗口，尾段 mask，按长度加权。[论文](https://arxiv.org/abs/2502.05139) [代码](https://github.com/facebookresearch/audiobox-aesthetics) | 官方 code+weights 公开，主体 CC-BY 4.0、部分 WavLM MIT；Py≥3.9/PyTorch≥2.2，自动 CUDA/MPS/CPU。[权重](https://huggingface.co/facebook/audiobox-aesthetics) [推理实现](https://github.com/facebookresearch/audiobox-aesthetics/blob/main/src/audiobox_aesthetics/infer.py) | **优先广谱质量 guardrail**。不是缺陷专用；16k前端看不到8k以上原始细节，mono丢立体声。可能偏好明亮/饱满/高制作感。 |
| **PAM** | 无参考；MS-CLAP 2023 对 clear/clean 与 noisy/artifacts 两个固定反义 prompt 的温度 softmax，取 clean 项，约0–1；不是校准缺陷概率。原实现44.1k、7s窗口。[论文](https://arxiv.org/abs/2402.00282) [代码](https://github.com/soham97/PAM) | PAM code MIT；依赖官方 MS-CLAP weights 的 MS-PL，不可把整体都称MIT。[权重](https://huggingface.co/microsoft/msclap) | **优先探索性 artifact feature**。作者评估含TTM，但“电音”具体缺陷未验证。量化自然度不可直接沿用softmax含义。 |
| **冻结音频 embedding + 本域小 head** | 同prompt/相同内容偏好对，专门问“不想要的人工感”；冻结encoder，线性/小MLP输出标量；Bradley–Terry偏好loss；DSP作辅助特征而非真值。 | 可用本轮已缓存的encoder；从头训练head。MuQ code MIT、weights CC-BY-NC 4.0；若商业用途需选可用encoder或取得许可。[MuQ](https://github.com/tencent-ailab/MuQ) [权重许可](https://github.com/tencent-ailab/MuQ/blob/main/LICENSE_weights) | **最有希望的主reward路线，尚需人工标签**。必须按prompt/歌曲/模型隔离划分，防止同源切片泄漏。正常合成器作显式负误伤对照。 |
| **MuQ-Eval A1** | 无参考，24k/10s，冻结MuQ310M+attention+小head，输出MI/TA约1–5；audio-only TA不能真正核对输入文本。[论文](https://arxiv.org/abs/2603.22677) [代码](https://github.com/dgtql/MuQ-Eval) | head仓库声称MIT；底层MuQ weights仍NC；[A1权重](https://huggingface.co/zhudi2825/MuQ-Eval-A1)。作者报告GPU实时，不能转写成本机速度。 | 后续质量对照；偏“整体生成音乐质量”，不是人工感专头。其同数据集CV与本域新模型泛化应分开。 |
| **SongEval** | 无参考；实际API五头依次为`Coherence, Musicality, Memorability, Clarity, Naturalness`，24k mono→MuQ第6隐藏层，`tanh*2+3`输出(1,5)。论文/README将Naturalness定义为vocal breathing&phrasing、Clarity为song structure；概念解释不能替代API字段。[论文](https://arxiv.org/abs/2505.10793) [API](https://github.com/ASLP-lab/SongEval/blob/main/eval.py) [输出实现](https://github.com/ASLP-lab/SongEval/blob/main/model.py) | 公开工具与checkpoint；项目CC-BY-NC-SA4.0，依赖MuQ。CPU显式支持，官方说明很慢。 | 沿用现有独立音乐性评估；“vocal naturalness”不等于混合音乐全局人工感。不能强迫纯器乐出现人声。 |
| **MuseCritic** | song+rubric，先生成critique，再输出五维(1,5)分数，含vocal naturalness；长歌曲评估与Muse-GRPO实现。[论文](https://arxiv.org/abs/2608.11755) [代码](https://github.com/WuqnEl/MuseCritic) | [官方权重卡](https://huggingface.co/WuqnEl/MuseCritic)标Apache2；完整依赖仍逐项核查。官方推理示例CUDA、可生成4096token，成本大于小head。 | 沿用已有独立arm。适合解释/复核，当前没有证据能可靠区分正常电子音色与不自然伪影。 |
| **CMI-RM** | candidate audio + 可选text/lyrics/reference-audio；24k mono，默认30s分块后拼embedding；输出alignment/quality，benchmark contract列为alignment/musicality。[论文](https://arxiv.org/abs/2603.00610) [API](https://github.com/Haiwen-Xia/CMI-RewardBench/blob/main/baselines/inference.py) | toolkit Apache2但README徽章NC-SA；[权重卡](https://huggingface.co/HaiwenXia/CMI-RM)明确CC-BY-NC4.0。不可从toolkit许可推导weights许可。 | 沿用已有arm，主要整体音乐性/条件对齐。0.7B F32，成本应实测；非人工感专用。 |
| **SongBench** | 七轴vocal/instrument/melody/structure/arrangement/mixing/musicality；instrument最接近乐器合成保真，仍含合法合成器语义。[论文](https://arxiv.org/abs/2604.25937) [代码](https://github.com/Tencent/SongBench) | 官方代码/模型公开，但[LICENSE](https://github.com/Tencent/SongBench/blob/main/LICENSE.txt)只允许academic，禁止commercial或production。 | 沿用已有独立arm作诊断；不能写成可直接生产RL部署。 |
| **TuneJury** | 单个audio+可选prompt输出偏好标量；冻结LAION-CLAP-Music+MERT330M，小head约2.8M；pairwise偏好训练，instrumental域。[论文](https://arxiv.org/abs/2606.17006) [代码](https://github.com/yonghyunk1m/TuneJury) | [官方卡](https://huggingface.co/TuneJury/tunejury)：主head CC-BY-NC4；CLAP-only A1变体Apache2；encoder首次下载约2.2GB，MERT gated。 | 值得后续独立偏好指标；非伪影专用。作者也报告强reward优化可牺牲fidelity，支持保留独立听测。 |
| **SingMOS / SingMOS-Pro** | 无参考、16k dry singing waveform→MOS，针对歌声质量；不应将整首混音当dry-vocal域。[论文](https://arxiv.org/abs/2510.01812) [代码](https://github.com/South-Twilight/SingMOS) | code MIT，torch.hub checkpoints公开；背后wav2vec2/s3prl与权重条款需保留。[官方数据](https://huggingface.co/datasets/TangRain/SingMOS)标CC-BY4。 | 若有真实vocal stem可加人声对照；先分离再评分会混入分离伪影。 |
| **UTMOS** | 无参考、speech MOS，VoiceMOS/Blizzard/VC域。[论文](https://arxiv.org/abs/2204.02152) [代码](https://github.com/sarulab-speech/UTMOS22) | 官方code MIT，strong checkpoints公开，fairseq依赖较旧。 | 语音自然度域外对照；音乐可能被当噪声；不推荐主reward。 |
| **DNSMOS P.835** | 无参考、16k noisy speech，约9.01s窗口；SIG/BAK/OVRL，面向噪声抑制。[论文](https://arxiv.org/abs/2110.01763) [代码与ONNX](https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS) | 仓库CC-BY4，公开ONNX，本地CPU实用；不需要云API。 | 背景音乐本来可能是需要抑制的内容，最易误伤音乐；只做反例或speech stem检查。 |
| **NISQA / NISQA-TTS** | 无参考，通信语音的quality/noisiness/coloration/discontinuity/loudness；TTS checkpoint才预测合成语音naturalness。[论文](https://arxiv.org/abs/2104.09494) [代码](https://github.com/gabrielmittag/NISQA) | code MIT；[weights CC-BY-NC-SA4](https://github.com/gabrielmittag/NISQA/blob/master/weights/LICENSE_model_weights)。公开本地PyTorch。 | 域不匹配，不能因名字有naturalness而应用混合音乐；不优先跑。 |
| **Distill-MOS** | 无参考，英语speech质量MOS，7.68s/分片；压缩SSL质量模型。[官方model card+code](https://github.com/microsoft/Distill-MOS) | MIT，公开weights，轻量本地路径。 | 比大ALM便宜，但仍是speech域；不解决风格与伪影分离。 |
| **FAD / FADtk** | 需要参考**分布**，不是“完全无参考”；embedding分布距离，也有per-song outlier模式。[论文](https://arxiv.org/abs/2311.01616) [官方工具](https://github.com/microsoft/fadtk) | tool MIT，encoder各有许可；CPU/GPU，需真实reference集。 | 适合整体分布与多样性审计，不直接等于每条自然度。参考集风格不匹配会惩罚正常电子乐。 |
| **ViSQOL Audio** | 完全参考：同内容clean/degraded，48k mono，spectrotemporal相似→MOS-LQO约1–4.75。[官方代码/指南](https://github.com/google/visqol) | Apache2，C++/Python，Mac可本地build。 | 适合codec重建/人为退化对；不能把另一首自然音乐作为reference衡量生成曲。 |

## 合成来源检测为什么不直接作为自然度reward

[SONICS 原论文](https://arxiv.org/abs/2408.14080)与[官方代码](https://github.com/awsaf49/sonics)提供 SpecTTTra 与公开checkpoint，目标是human-vs-AI来源分类。code MIT，但[数据许可](https://github.com/awsaf49/sonics/blob/main/LICENSE)另有NC与禁止用于Suno/Udio竞争产品研究条款；checkpoint还需逐卡核对。公开推理不等于本轮已运行。SONICS的48GB GPU复现配置不应写成单条推理最低需求。

[MusicDET 论文](https://proceedings.mlr.press/v306/han26u.html)与[MIT代码](https://github.com/Chaolei98/MusicDET)通过normalizing flow拟合真实音乐特征分布，产生OOD/合成检测信号。公开pretrained目录存在，所选权重和backbone仍需单独查条款；本轮不优先运行。

两个方法都不直接回答“听起来有多不自然”。高质量AI歌曲可能带不可闻来源指纹；合法但少见的电子音色可能离真实训练分布很远。将`1-p(AI)`当reward会鼓励隐藏来源或改变谱指纹，而非消除人耳讨厌的缺陷。只有先在本域盲听标签上证实其增量预测能力，才作为head辅助feature。

## 可解释DSP指标的含义和误伤

以下是特征定义与可检验假设，不是已经验证的自然度规律。

| 特征 | 测量和可能关联 | 容易误伤/被取巧 |
|---|---|---|
| clipping/plateau与真峰值 | 接近数字满幅样本比例、平顶run、过采样true peak；定位削波/过载 | 降低增益便可清零满幅比，模拟失真仍可能存在；正常限幅/吉他饱和不一定缺陷。必须保留原始增益与归一化双视图。 |
| 高频能量/centroid/rolloff | 频谱质量的高频占比、频率重心和累计能量边界；可能发现尖锐噪声或过度低通 | 镲片、明亮原声、齿音、synth lead正常；低通能“修好”分数同时毁清晰度。见[centroid定义](https://librosa.org/doc/0.11.0/generated/librosa.feature.spectral_centroid.html)、[rolloff定义](https://librosa.org/doc/0.11.0/generated/librosa.feature.spectral_rolloff.html)。 |
| spectral flatness | 几何均值/算术均值，测noise-like vs tone-like；可追踪嘶声/噪声化 | 高flatness的cymbal/shaker/气声正常；低flatness的buzz/正弦也会假高分。[官方定义](https://librosa.org/doc/0.11.0/generated/librosa.feature.spectral_flatness.html) |
| 谱峰显著度/固定窄带持久性 | 稳定频率峰超过局部谱包络、跨时间不动；可探索decoder固定buzz | drone/organ/synth pad本就稳定。应寻找与pitch/和声无关的峰、与codec stride一致的频率；没有decoder信息只可列假设。 |
| 谐波/非谐波、F0/包络连续性 | voiced stem上的harmonic deviation、谱包络跳变，可能发现vocoder异常 | polyphonic mixture的单F0估计不成立；bell/gamelan/piano非谐波合法。禁止在整曲上把低HNR当电音。 |
| 高频包络调制与周期峰 | 子带包络FFT、自相关，观测buzz/pumping/周期伪影 | tremolo、vibrato、节拍、sidechain为正确音乐；不能一概惩罚modulation。 |
| flux/onset与瞬态峰度 | 正谱差、起音密度/陡峭度，可探索过度平滑或破碎 | ambient/pad起音少正常，drum&bass起音密正常；可添加clicks刷分。[flux官方定义](https://librosa.org/doc/0.11.0/generated/librosa.onset.onset_strength.html) |
| crest factor/短时RMS分位差 | 峰值与RMS比、时间动态，发现强压缩、平滑 | 电子舞曲强压缩可正确；加入稀疏尖峰可刷动态。与整体音质/节奏同时判断。 |
| stereo correlation/side energy | 左右声道相干、mono折叠后能量变化，发现相位抵消 | widening/chorus/reverb是合法效果；mono模型无能力判断空间质量。 |

## 建议可执行reward定义

1. **标签先定义**：缺陷强度0–4（0无、4严重），整体音质1–5，音乐性1–5，意图符合/不符合/不确定；另记正常电子乐器标签。成对问题：“在同等音乐内容下，哪一个有更少不想要的人工感？”允许tie/无法判断。收集的是真实听者值，自动分数不回填人工列。
2. **主head**：`u(x,p)=MLP([frozen_embedding(x), optional_prompt_embedding(p), DSP(x)])`，损失 `-log sigmoid(u(preferred)-u(other))`；可加跨prompt风格均衡与同源增强不变性正则。至少按歌曲/prompt组划分训练/校准/测试，进一步留出生成模型。重复切片不得跨集合。
3. **校准**：在独立人工集用logistic/isotonic标定成“不想要人工感≤1”的概率；无标签时只报原始score或固定校准集分位，不称probability。PAM softmax既不是自然度概率，也不是人工偏好概率。
4. **片段聚合**：保留全曲覆盖；当前接口示例取10s窗口、5s hop、均值与Q20。等长窗口不按能量加权，防止大音量掩盖小音量缺陷；上线前还需明确尾部短段mask/时长权重，不能暗称现有接口已实现。`r_nat=.7*mean(r_i)+.3*quantile(r_i,.2)`，系数为待验证起点。另报最差3片段给听者。纯静音/损坏音频拒绝评分，正常休止片段不当缺陷；整曲低能量拒绝与否取决于prompt/片段用途。
5. **组合**：若现有总reward已z标准化，`R=R_existing + λ*clip(z_cal(r_nat),-2,2) - μ*max(0,q_floor-q) - ν*max(0,m_floor-m)`。`q=Audiobox_PQ`，`m=既有独立musicality`；floor用SFT/baseline的条件分布预先设定。λ先shadow=0，验证后小规模扫0.02/0.05/0.1，不写成已选最优。μ/ν用约束违反率选择，不能假装已有标定。KL保持现有方案。无主head标签时，PAM/PQ只保留独立字段；不要悄悄把PQ改名nat。
6. **部署**：冻结encoder缓存只适合固定音频；on-policy新采样必须重新算。异步reward worker、batch、hash cache、明确scorer版本/校准版本；超时/NaN返回invalid而非0.5奖励。训练计算不能反传通过`inference_mode` scorer；policy梯度仍走RL。

## 最小反例与评估方案

同一原始片段派生 gain−20/−40dB、纯静音、低通1.5/4k、高架EQ、压缩、hard clip/soft saturation、轻重bitcrush、固定buzz与ring modulation；保留正常原生synth、明亮acoustic、cymbal作为受保护对照。它们是已知处理，不是缺陷强度人工标签。低通/压缩可能在某些音乐中被偏好，不能强制声明人为退化必差。

在真实人工标签到位前，只报变换响应、相对原片变化、受保护对照误伤和吞吐。人工到位后报group bootstrap的Spearman（方向统一）、pairwise accuracy（允许ties）、模型/风格子组、quality/musicality trade-off；样本少则提供置信区间且标探索性。不能用这些伪变换的“expected label”计算并宣称对人类一致性。

## 与音乐RL和主观评价直接相关的依据

- [MusicRL 论文](https://arxiv.org/abs/2402.04229)与[作者试听页](https://google-research.github.io/seanet/musiclm/rlhf/)：质量、文本符合度和大规模偏好属于不同reward，不能由单一技术质量代替完整音乐偏好；未核实有可复用公开reward checkpoint。
- [Rubric-Based Optimization for Text-to-Music Generation](https://arxiv.org/abs/2610.03589)：同prompt候选配对、ALM rubric与多指标取舍是直接相关方法；论文报告单评估器优化可恶化独立评估器。这里借鉴设计，不等于本轮训练已有收益。
- [TuneJury](https://arxiv.org/abs/2606.17006)：冻结embedding+小head及校准可行；其instrumental域与本域不同，强优化出现fidelity取舍，不能拿作者指标作为本轮结果。
- [Generative Adversarial Post-Training](https://arxiv.org/abs/2511.17879)：melody-to-chord实时互动任务有coherence reward引起多样性下降的例子，支持监控多样性；符号和弦域证据不能直接证明音频人工感改善。
- [ViSQOL官方指南](https://github.com/google/visqol)：参考必须同内容且clean，不同音乐不能直接算保真；单个分数需同处理多样本聚合解释。

## 复现实验前应注意的实现问题

- PAM upstream `dataset.py`直接reshape stereo，会把左右声道拼接；自行先`mean(channels)`，再44.1k、7s；短片重复与尾部重叠策略必须显式记录。本地源码commit `ead6e7f9d907268e30c73b29526f50523a34b4aa`。
- PAM旧requirements含Windows与旧Torch pin，本机macOS不照单安装，固定实际工作依赖并记录改动。
- Audiobox upstream commit `2618e9d451b456e9328b39495b5e6234678aa550`，模型会自动选MPS；必须测该设备路径，不能把CUDA作者数值搬来。
- MuQ-Eval本地审阅commit `60a88f8ac0909ca1fd1a78af3660f1fc376977a1`；README示例漏导入AudioProcessor、model card与v2论文对degradation验证描述未同步；采用具体代码和本机输出作复现依据。
- “源码公开”“权重公开”“训练数据公开”“许可允许商业生产”分别记录。SongEval/SongBench/CMI-RM/MuQ/TuneJury限制不同，不能统一写成“开源可直接用于RL”。
