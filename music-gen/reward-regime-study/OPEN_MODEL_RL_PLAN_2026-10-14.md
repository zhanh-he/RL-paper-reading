# 开放歌曲模型的 RL 路线与质量分层实验

**状态更新：用户已正式授权启动。DiffRhythm2 已有真实优化更新；HeartMuLa / SongBloom 的准备已成功，六个GPU任务已排队。核验日期：2026-10-14 AWST。**

## 本次执行变更

- 启动证据、任务ID、版本、边界见[执行记录](runs/2026-10-14-open-head-startup/README.md)。后文保留先前设计推导，不把“拟实现”字样当作最新状态。
- 今晚首轮选择**受限输出头**：HeartMuLa 的 codebook0_head、DiffRhythm2 的 DiT proj_out、SongBloom 的最终 sketch classifier；均 rank8/alpha16、LR3e-4、270步、G2、beta0.01，原生60秒预算。它们不是完整backbone微调，不能用阴性结果断言某模型无法RL。
- DiffRhythm2 使用官方Flow-GRPO SDE helper，31次随机转移/block，重算完整block高斯log-prob；冻结上游特征后，输出头的条件概率可精确重算。首步重算差0、参考KL0、梯度非零已核验。KL是按latent维度归一化的条件高斯KL；不称波形KL或与token KL同单位。主要baseline是同配置SDE；ODE部署迁移对照仍未完成。
- SongBloom 使用社区恢复的150s base，不是DPO版；作者原端点仍不可独立匹配。365个重复VAE别名仅在逐tensor完全相等后删除，其余strict加载通过。固定300条reference的哈希与解码检查已通过，不公开源音频。
- 每个文字条件实验固定五个试听case：原valid0/10/20，加同valid0歌词、同seed5101的2类/5类乐器改写探针。原三首不额外限制数量。改写探针不进入正式train/30valid/test，也不将自动reward解释为真实乐器计数。
- SongBloom不吃Text，因此数量提示遵循为n/a；改用五个独立原生条件0/10/20/1/2，避免重复音频充数。原始Text仍展示，并注明未输入模型。
- 无定时任务、无夜间Codex监控。有限训练、验证、推理脚本继续自行执行，等待用户手动回收。

本文保留新增实验技术路线，不替换已有 [六臂协议](protocol.json)，也不是训练完成记录。后续用户已确认HeartMuLa / DiffRhythm2 / SongBloom名单并移除SongGen，ACE-Step1.5留在最后；新增模型仍需完成各自真实训练接入，尚未提交。已有结果继续按原协议归档，不把旧短片段结果冒充下面的新对照。

## 1. 建议先决定什么

1. 把问题定为：**不同起点质量的可训练歌曲模型，分别优化 SP / AI reward 后，改善的是听感、任务能力，还是只改善代理分数？**
2. 优先模型为 **YuE2、DiffRhythm2、SongBloom**；**HeartMuLa** 补充中间质量段，已有 **Muse** 保留为工程与历史对照。不再人为劣化强模型来替代真实的较弱模型。
3. 每模型保留三个独立目标：**SP Richness-v1 / SongBench-Instrument / SongBench-Arrangement**。首轮只用 **LR 3e-4**，不扩 grid search。SongEval 用于独立评估，它没有 instrument 子分。
4. **AR 用 token GRPO，连续 flow 用有明确转移概率的 Flow-GRPO**。SongBloom 另有低成本但能力受限的 sketch-head 路线，不能把两者混称同一个实验。
5. 优先做 DiffRhythm2 的声学 RL 接入；SongBloom 同时解决版本、权重与参考音频问题。名单已确认，按最新请求先整理online demo；未实现的适配不冒充已启动。没有定时任务。

## 2. 模型名单与开放状态

下表 MOS 来自 **Zhanh 转述的同事内部快速听评：10 个相同输入，每模型 10 首**。尚未收到逐条评分、盲测方法、人数、生成设置和具体 checkpoint。这些数字用于提出质量分层假设，不是公开 benchmark、我们重测的 MOS，或模型家族的固有分数。

| 模型 | 团队快听 MOS | 当前核实情况 | 本研究角色 |
| --- | ---: | --- | --- |
| YuE2 | 约 4.2 | [代码](https://github.com/multimodal-art-projection/YuE)、[3B 权重](https://huggingface.co/m-a-p/YuE2-3B)公开；沿用已记录的研究使用条件和确切版本 | 高起点锚点；已有 GRPO 可复用工程，但不自动复用不同协议的分数 |
| HeartMuLa | 约 3.5 | [官方代码与权重入口](https://github.com/HeartMuLa/heartlib)公开，Apache-2.0；有原始 3B、官方 RL 版和 happy-new-year 版，不能混为一个起点 | 中间质量段；需要新建分层 codec-token log-prob 接口 |
| DiffRhythm2 | 约 3.0 | [论文对应仓库](https://github.com/xiaomi-research/diffrhythm2)、[ASLP 发布分支](https://github.com/ASLP-lab/DiffRhythm2)、[权重](https://huggingface.co/ASLP-lab/DiffRhythm2)可访问；代码和权重为 Apache-2.0 | 首要连续 flow 实验。**不是 DiffRhythm v1 / v1.2 / DiffRhythm+** |
| SongBloom | 约 3.0 | 曾公开推理代码和权重；本次两个原代码入口返回 404，作者原 HF 权重入口返回 401。社区副本可访问，来源和许可证需随具体版本核验 | 保留重点模型，但当前不是“随时可启动”的状态 |
| Muse | 本次未提供 | [代码](https://github.com/yuhui1038/Muse)及此前实验工程已有记录 | 补充对照；不擅自填一个 MOS，也不以它取代 DiffRhythm2 / SongBloom |
| Suno 4.5 | 约 4.0-4.2 | 此处仅保留团队已有试听对照；没有可用于本研究本地梯度训练的权重 | 不安排 actor RL；保存版本与既有音频来源，不冒充本地训练 |
| “盘古 / Mureka” | 约 4.0-4.2 | 名称沿用用户口述，具体产品和版本待确认，不自行认定两者是同一模型 | 同上，仅作已有外部试听参照 |
| Private baseline | 不公开 | 不可获取的内部模型；公开页面仅保留通用占位，不披露内部信息或索取文件 | 不纳入可执行RL矩阵，不填入替代音频、分数或曲线 |

**SongBloom 的“开放”要说准确：**社区代码副本中的 [LICENSE](https://github.com/Shamdon/songbloom/blob/990987ec94c49056a8dbf838987af93a37a1250d/LICENSE)限制学术用途；某些 HF 副本却标 Apache-2.0。这里不能凭副本标签宣称原权重无限制开放。先归档拟使用版本的来源与随附许可，按已确认的学术研究范围准备；目前没有下载或执行副本权重。404 / 401 只说明本次访问失败，不证明永久撤回或不能向作者获取。

### 拟固定的版本

| 模型 | 候选起点 | 版本冻结要求 |
| --- | --- | --- |
| YuE2 | 当前研究使用的原始 3B，而非已 SongEval / MuseCritic 对齐的 adapter | 复用原 run manifest 中模型、tokenizer、codec、规划配置与 SHA；不能悄悄更换 decoder |
| HeartMuLa | 默认候选 `HeartMuLa-oss-3B-happy-new-year` + `HeartCodec-oss-20260123` | 先核对团队 3.5 分究竟来自哪个版本。候选权重 revision 为 `41f6fc68490e11dc43fdabaa6b5767946408c903`，不是已完成下载或性能验收 |
| DiffRhythm2 | `ASLP-lab/DiffRhythm2` 的 `model.safetensors` + 同 revision decoder/config | 候选 HF revision `9aa15742e4889c0eb2e198db6fdab1facf1b6761`；发布的 checkpoint 可能已有论文所述偏好优化，不能称未经后训练的 base |
| SongBloom | 首选与团队试听一致的 checkpoint；否则先登记 `songbloom_full_150s` 为候选 | 明确 base / `150s_dpo` / `240s`，不跨版本移用 MOS。社区候选 [rsxdalv 副本](https://huggingface.co/rsxdalv/SongBloom/tree/093351709a0efa263569cc4b679429d38f4508ec)仅作恢复线索 |

模型起点是否已做过 DPO / RL 必须在页面显示。我们的结论是“对这个发布 checkpoint 继续后训练”，不是“从无偏好训练的底座首次做 RL”。

## 3. 数据不再重新切

沿用本研究已冻结的 [CMI-Pref 三模态来源协议](../lyrics2song/rl/grpo/runs/2026-10-06-cmi-triple-9to1/README.md)和 [可下载 split JSON](../../platform/site/demos/data/cmi-pref-triple-9to1-split.json)。此前聊天称 CMI-Perf，这里的实际来源名为 [CMI-Pref](https://huggingface.co/datasets/HaiwenXia/cmi-pref)。

| 用途 | 固定成员 | 能做什么 |
| --- | --- | --- |
| Train | 270 个唯一条件 | 在线生成候选并计算 reward；不是用现成偏好对假装在线 GRPO |
| Valid | 30 个唯一条件 | 全量验证、检查失败/退化、确定 checkpoint；不进入梯度 |
| Primary external test | WildSongBench 全部 192 个条件 | 方案冻结后评估；不按 test 分数调 reward、LR 或挑模型版本 |
| 另存的 CMI official test | 121 个唯一条件，来自 125 票 | 仍封存；如需增加同域测试，单独确认，不混进这 30 个 valid |
| 同事的 10 个快听输入 | 尚未交付成员清单 | 仅作为已有观察与可行性线索，不替代正式数据或全量 test |

原始 train 中“三字段非空”的 **469 是偏好票数，不是 469 个独立训练条件**：先得到 325 个条件，排除与官方 test / WSB 规范化歌词重合后剩 300，再按歌词组稳定拆为 270 / 30。新模型不能各自随机重切或静默剔除难样本。

Split 文件 SHA256：`db769e27d75a70f2e819d6de771002652bb37e9adeb4a549fca66d9dfa30c616`。私有三模态 manifest SHA256：`2f5f31219514ad479984e921a8dc6603680e275f31531778c7bb018da5df0136`。

**新增的参考音频验收：**旧 YuE2 / Muse 实验没有下载、解码或使用这批 reference audio。字段非空不等于音频已下载且可读。SongBloom 接入前须逐项检查固定 300 条的资源可获取性、原始 SHA、时长与解码；任何缺失都报告准确数量和 condition ID，不另造一个“全量”子集。保留原 split，缺失处理须审阅。

### 默认输入与公平性

| 模型 | 本轮拟使用的原生输入 | 是否使用 CMI reference audio |
| --- | --- | --- |
| YuE2 | 现有 native text/tags + lyrics 路径；规划方式固定 | 否 |
| HeartMuLa | text 转 tags + lyrics，转换规则固定 | 否；当前公开 pipeline 遇到 `ref_audio` 明确抛 `NotImplementedError` |
| DiffRhythm2 | lyrics + MuQ-MuLan text-style embedding | 否；发布代码允许 style 字段为文字或音频路径，首轮固定文字分支 |
| SongBloom | structured lyrics + 10 秒 reference audio | 是；不假装已经拿到文字条件版的权重 |

因此，**主结果先比较同一个模型内部 baseline -> GRPO 的配对变化**。跨模型差异还混合了架构、输入模态、预训练与可训练参数位置，不能全归因于 MOS。需要真正的同模态因果对照时，另设一致条件轨道，不把不同输入包装成完全公平的总榜。歌词格式化只能改结构/分隔符，保存原文与转换记录，不改写歌词内容。

**WSB 的 SongBloom 特例：**[官方协议](https://huggingface.co/datasets/m-a-p/WildSongBench/blob/e361b85a8d7365d3079fa9647641a87ce3c8c5b5/benchmark/results.md)记录 SongBloom 使用固定 audio prompt，而 [公开数据](https://huggingface.co/datasets/m-a-p/WildSongBench)不包含音频。先确认能否取得该参考文件及其 SHA；否则建议在打开本研究 test 输出前，从有使用权的 train 参考资源固定一个 10 秒片段，所有 192 条、所有 reward 臂共用，标为 **WSB / fixed-reference adaptation**。它不是原论文音频配置的复现，也没有使用 WSB 目标音频。无法建立可复现输入时标 `n/a: missing reference protocol`，不能暗中拿目标歌曲或最优输出当 reference。

## 4. 奖励定义与要验证的 badcase

三个单目标臂沿用 [已实现 reward](rewards.py)和 [版本化协议](protocol.json)，固定 Demucs、SongBench 与音频前端。所有候选使用同一份 immutable lossless 音频供各 scorer 打分，保留原始分数与公共 guard，不拿 MP3 再评分来替代原结果。

| 目标 | 定义 | 实际含义 | 需要主动检查的失败方式 |
| --- | --- | --- | --- |
| SP Richness-v1 | `G * (0.60*activity + 0.25*movement + 0.15*active_fraction) - P` | 冻结分离器得到伴奏后，统计分频活动/变化并加 tonality gate；不是乐器数量 | 噪声铺底、持续密集音、无意义音符堆叠、伴奏淹没人声、分离泄漏 |
| SongBench-Instrument | `Instrumental / 10 - P` | [官方含义](https://github.com/Tencent/SongBench)：乐器逼真度、合成与声学保真 | 干净但单一的音色、高保真循环，分数涨而编配/歌词/多样性下降 |
| SongBench-Arrangement | `Arrangement / 10 - P` | 和声框架及配器编排的艺术性，比 Instrument 更接近编配问题 | 套路化和声、过密配器、牺牲主唱、只对短窗口讨好 scorer |

`P` 只减一次：现有削波与 RMS 公共 penalty。无效、非有限、极短/近静音候选按现有规则处理，同时报告发生率。具体阈值沿用 `protocol.json`，不在新模型中偷偷调 reward 口径。KL 是独立的 loss 项，不算进此表的目标分数。

合成攻击仅用于检查 reward 的不变性与漏洞，不冒充模型 GRPO 结果。`target↑ + another scorer↓` 先标 **proxy conflict**；经盲听确认听感/任务质量下降后，才标为有证据的 reward hacking。单一 scorer 的高分不是感知真值。

**组合 reward 是第二阶段，不挤占第一阶段单目标结果：**候选为 `R_mix = (SP + Instrumental/10 + Arrangement/10)/3 - P`，同一 guard 只算一次，权重预先固定。先在 DiffRhythm2 / SongBloom 各增加一臂，与三个单目标比较。组合不是天然防 hacking：两项 AI 来自同一模型，可能共同失效；用独立 SongEval、歌词指标与盲听判断，不用它自己的总分证明自己有效。尚未授权启动。

## 5. 分模型 RL 工程路线

以下都是**拟实现方案与验收条件**，不是声称已有可运行 trainer。公开推理代码和可读权重意味着有适配基础，不等于作者已经提供 GRPO 训练脚本。

### 5.1 YuE2：保留已跑通的离散策略

沿用 PyTorch / PEFT 的 native semantic-token GRPO、冻结 decoder 与 reference。采样和重算必须使用同一 codec/EOS 支持集、最短长度规则、温度和 CFG/规划设置。逐步记录 target、guard、有效组、梯度、policy/reference 身份与 sampled KL。

已有短预算六臂结果仍属于 v3。若本计划改成长预算，须新建 run ID 和自己的 step-0，不能把旧 baseline 或 reward 曲线接过来；已完成训练不会因为这份提案被自动重跑。

### 5.2 HeartMuLa：global / local 两层 token likelihood

已检查 [源码](https://github.com/HeartMuLa/heartlib/tree/5858ca8f1ffe58d62be7c007ea55f1033c5ca456/src/heartlib)：`generate_frame()` 先生成 codebook-0，再逐个生成其余码本；当前模型文件没有开箱即用的训练 `forward(labels)`。需要增补 teacher-forcing scorer，而不是直接把 HF 模型名塞给 TRL。

拟先在 global backbone 加 LoRA，冻结 local decoder 和 HeartCodec 参数。**冻结 local 权重不代表其输出概率与 actor 无关**：local logits 仍接收 global hidden state。必须记录并重算每帧所有被采样码本的条件 log-prob，含 EOS/mask、CFG 和真实历史，不能只算 codebook-0 后声称是完整 policy GRPO。

训练态 dropout、KV cache、top-k 截断必须与 rollout 对齐；优先使用无 top-k 的全支持训练分布，native 试听配置另列。先验证未更新前 old/current ratio 和 reference identity，再验证 LoRA 有限且非零梯度。codec 固定精度、随机状态与 chunking，不能把 renderer 变化计成 RL 收益。

### 5.3 DiffRhythm2：block Flow-GRPO 是重点适配

[论文](https://arxiv.org/abs/2510.22950)采用 block flow，而发布的 [CFM sampler](https://github.com/ASLP-lab/DiffRhythm2/blob/7804f821b797b4f276090e1a9dcd37e97d9915d5/diffrhythm2/cfm.py)是随机初始 latent 加确定性 ODE，`sample_block_cache()` 被 `no_grad` 包裹。它没有现成的 policy transition log-prob。论文的 **CPPO 指 Cross-Pair Preference Optimization，不是已发布的 online GRPO trainer**。

拟借鉴 [官方 Flow-GRPO](https://github.com/yifan123/flow_grpo)而非重写基础概率算法，适配其连续随机转移到 native DiT；冻结 MuQ-MuLan、歌词前端与 decoder，仅在 DiT attention 加 LoRA。该框架目前不是 DiffRhythm2 开箱适配，仍需以下接入：

1. 核对 `t=0 noise -> t=1 data` 的速度符号、噪声日程与 CFG：DiffRhythm2 与 SongBloom 的时间方向相反，不能复制同一符号。
2. 用有方差的 SDE 转移采样，保存 block / denoise time、state、next-state、old log-prob、history 和噪声参数；奖励在解码后计算，不要求 SongBench 可微。
3. 更新时在相同 detached state/history 上重算 current/reference 转移。不能把含旧参数特征的 KV cache 当成新 actor 的当前概率；先做 no-cache 对照，再考虑等价缓存加速。
4. 同方差高斯转移的条件 KL 可解析计算；同时保存明确的求和和按有效 latent/step 归一化版本。若只训练部分 SDE 时间点，标 **window transition KL**，不称完整轨迹或波形 KL。
5. 排除零方差端点的虚假 log-prob；实际 NFE、SDE window、噪声强度、方差与 loss reduction 全写 manifest。首版不套 Fast/Guard 等多种新变体，先把标准适配核验清楚。

**重要控制：**SDE 化本身可能改变质量。必须保存原生 ODE baseline，以及零 adapter、相同 SDE 配置的 RL baseline；主要配对是 `base-SDE -> GRPO-SDE`，另报 `base-ODE -> GRPO-ODE` 的部署迁移。不能只拿 ODE base 对比 SDE RL 来证明训练有效。

官方推理另有默认 fake-stereo 的后处理。本实验拟禁用它，直接在原始 decoder 输出评分，baseline 与 GRPO 同处理；页面显示真实声道数。这是实验控制，不把后处理变化算成模型提升。

### 5.4 SongBloom：先分清两条可做路线

[论文 §4](https://arxiv.org/html/2506.07634#S4)中 sketch token 与声学 block 交错：AR hidden state 进入 flow，声学历史又反馈 AR。因此不能简单宣称“冻结 flow 权重，所以只用 sketch log-prob 就能正确训练整个 AR backbone”。改变 hidden state 仍会改变声学生成分布。

**路线 S1：sketch-head-only GRPO，建议最快的首轮验证。**冻结整个 AR transformer、conditioner、sketch embeddings、flow、VAE 与 acoustic history encoder，仅对离散 `skeleton_classifier` 的 Linear 加 LoRA。此时模型参数只改变离散动作分布，冻结 renderer/context 作为环境；按采样的 sketch 动作及真实交错声学历史重算概率。不能把声学历史丢掉后做普通纯 token teacher forcing。此方案不要求对 ODE 凭空定义 Gaussian log-prob，也不通过 reward 做反传。

S1 结果严格命名 **SongBloom / sketch-head GRPO**。它测试“有限规划动作是否能通过 RL 改善”，不是充分测试“整个弱生成器能否改善音色”。没有收益时，不能下结论说 SongBloom 无法 RL。

**路线 S2：acoustic Flow-GRPO，建议作为声学主结论。**冻结 AR 与条件模块，对 NAR DiT 加 LoRA；参考 DiffRhythm2 的 SDE 转移接口，按 SongBloom 自己的 `t=1 noise -> t=0 data` 重新适配。AR 依据已采样声学历史运行，记录完整上下文。VAE 与最后音频 reward 固定。完成 ODE/SDE baseline 控制后再训练；不能把 flow MSE 随手改名 KL。

S2 比 S1 更直接测试声音质量的可优化性，工程成本也更高。二者用独立 run ID，不能拼成同一条曲线；联合更新 AR 与 flow 留给后续，不在赶工首版里遗漏条件路径的概率导数。

**当前边界：**模型历史结构依据论文；社区 [代码副本](https://github.com/Shamdon/songbloom/tree/990987ec94c49056a8dbf838987af93a37a1250d)中可看到 `skeleton_classifier`、`diffusion_forward()`、Euler sampler 与训练 forward，但它不是本次成功访问的作者原仓库。上述接口名只作待审适配位置，正式实现前须确认副本与选定权重匹配。缺少权重/来源时报告阻塞，不提交假训练。

## 6. 实验矩阵与预算

| 阶段 | 模型与目标 | 规模 | 定位 |
| --- | --- | ---: | --- |
| 已批准 v3 | YuE2 / Muse × 三个单 reward | 6 臂 | 保留已有协议与进度，不因本提案重启或挪用为新结果 |
| 新计划 A | YuE2 / DiffRhythm2 / SongBloom × 三个单 reward | 9 臂 | 优先强/弱对照；SongBloom 路线 S1 或 S2 必须显式选择 |
| 新计划 B | HeartMuLa × 三个单 reward | 3 臂 | 补中间质量段；与 A 合计 12 臂，不是 12 个模型 |
| 后续 C | DiffRhythm2 / SongBloom × 预定义组合 reward | 2 臂 | 验证能否制衡单目标漏洞，不默认承诺有效 |

新 A 中如直接引用已有 YuE2 v3，只能作历史参照；正式等预算矩阵需要同预算 YuE2 对照，不把复用结果当成新增完成臂。若 SongBloom 先 S1 后 S2，多出的三臂是新实验，不能隐藏额外预算。

### 单一学习率的建议首轮配置

- `AdamW + LoRA, LR=3e-4, seed=42, 270 optimizer updates, group size=2`；第一轮不做 LR search，不用较弱模型偷偷享受更多 best-of-N。
- 每步一个 train condition、两个独立候选；固定 270 条覆盖与顺序，所有模型记录实际曝光 ID 与次数。零方差 reward 组记录为无相对优势，不伪造梯度信号。
- 首选每组一次更新，与现有 urgent 方案一致。第一次评估的 old/current ratio 为 1，**ratio clipping 在这一次梯度中没有实际约束作用**，不再把 clip 当成已生效的保护；固定 reference 正则与监控才是主要约束。
- 初始 reference 系数拟为 `beta=0.01`，沿用已有研究习惯；但 token KL 与连续 transition KL 的单位和 reduction 不同，不能因为 beta 相同就宣称约束强度一致。按架构报告单位与有效长度，出现剧烈漂移先停该臂排查，不暗中改 LR。
- LoRA 默认候选 `rank=8, alpha=16, dropout=0`；旧 YuE2 rank4 不冒充一致配置。具体 target modules、参数量与显存须在审批后的接入验收中固定；架构间同 rank / LR 也不代表同更新强度。

**推荐本次新矩阵使用 native 60 秒 rollout budget**，避免把只优化十几秒的目标误作长曲改善。保留自然提前结束，不补静音、不循环、不靠换采样率凑时长。训练 scorer 对同一候选按固定规则处理；若前端有裁剪限制，记录窗口起点/长度、短音频行为及聚合，baseline 与 GRPO 一致。结构指标在受支持的长上下文上另报，不能从首个短窗推出整曲结构收益。

这意味着新矩阵不会与旧 600-token YuE2 曲线直接等同。若急需更短训练预算，先明确版本化另一套协议，而不是实现时偷偷缩短。SongBloom 的 native 最小时长、歌词完整度、EOS 和 block 边界要一起验收；不强压成超出已核验用法的短片段。

以 270 步、每步 2 首估计，每臂 **540 次训练候选**；12 臂共 **6,480 次**。若每首达到 60 秒，仅训练候选就相当于 **108 小时输出音频**，不含 baseline、验证、重试与 scorer。这个是样本工作量，不是 GPU 墙钟 ETA。

审批后先用真实资源测得 rollout / decode / 三 reward 的时间、峰值显存和失败率，才估算完工时间。不能仅凭“1B/3B”或已用 5090 就承诺今天完成。接入验收不算正式训练结果，也不会代替全量验证。

## 7. 全量验证、独立评估与判断规则

**Valid：**固定全部 30 条，在 `0 / 50 / 100 / 270` 验证；训练中保存 `1 / 5 / 25` checkpoint 供固定示例回放，不把三个示例均值当 valid 曲线。每模型同 baseline、同条件、同种子、同 renderer 与时长规则。不同结构模型的 seed 只约束各自配对，不意味着采样到相同随机路径。

**Test：**方案和终点先冻结，默认全部预登记单 reward 臂用 step270，在全部 192 条 WSB 上作一次 final 评估；如发散未到270，显示失败及缺失数，不用中间最好 checkpoint 偷换。另报 best-valid 时须列选择规则和候选范围。当前不启动 Q3O，不做 SongBench-GRPO 之外的新整分目标；本矩阵优化的是明确的 Instrument / Arrangement 子项。

| 评估维度 | 必报内容 | 能否作为独立证据 |
| --- | --- | --- |
| 优化目标 | raw target、guard 后 objective、失败率、组内方差 | 只能证明代理目标变化 |
| 跨 reward | SP 全部组件、SongBench 七维与均值 | 未优化的分项有帮助，但同一 SongBench 各 head 不独立 |
| 外部 scorer | SongEval 五维及均值；按既有 WSB pipeline 加 MuQ-MuLan、AllMusicCaps、AudioBox PQ | 固定版本与可适用条件；未算出的指标注明 pending / n/a 原因 |
| 歌词 / 条件保持 | PER、ASR 空输出率、text alignment、reference 条件说明 | PER 对截断歌词/片段对齐规则必须明确，不能把整篇未唱歌词直接罚进短片段 |
| 声音与塌缩 | RMS、peak、near-fullscale、LUFS、有效时长、重复/多样性，分离器泄漏审计 | 不以单一 flatness 或高频能量断言音质 |
| 人类听评 | 盲 A/B、整体 MOS、乐器真实度、编配与歌词可懂度 | 核心感知证据；记录听者、量表、顺序随机化与置信区间 |

正式 WSB 的指标、候选选择与模型输入配置见 [官方复现协议](https://huggingface.co/datasets/m-a-p/WildSongBench)。本研究主分析先用固定 **single-sample** baseline / GRPO 配对，避免 reward 相关 reranking 混入训练效应；它不是标准 YuE2 two-candidate lower-PER selection 的数值复现。如需该系统级协议，另报所有臂一致的两候选结果与额外成本，不把两种数字混表。

统计以条件为配对单位，报告 delta、改善比例、分布和 condition-clustered bootstrap CI。第一轮一个训练 seed 是探索性结果；不能把跨30条条件的 CI 当作跨训练随机性的证据。多个 reward / 模型的显著性主张需校正或明确 exploratory。

“低 MOS 更适合 SP、高 MOS 更适合 AI”的检验看 **baseline quality × reward family 的交互**，不是看哪条自身目标涨得最多。十首快听的分层只能启发，不足以确立 3/4 分阈值。跨架构、模态与可训练模块的混杂必须保留；后续同家族不同质量 checkpoint 是更强的因果控制。

**研究心得与可支持的结论：**我们的贡献应是可复现的跨质量起点、跨策略类型 reward 对照，不宣称“首次给这些音乐模型做 RL”，因为部分发布模型已有 DPO / RL。出现 `target↑ + 盲听↑` 才支持感知改善；`target↑ + 盲听↓` 支持过度优化风险；`target不变 + 组内方差很小` 首先是奖励分辨率/探索问题；`梯度或解码失败` 是工程失败，不能拿来证明弱模型不可优化。SongBloom / DiffRhythm2 的相近快听 MOS 并不意味着两者会有相同学习曲线。

## 8. Demo 与 badcase 的呈现

每个实验命名为 `DiffRhythm2-SongBench-Arrangement-3e-4` 等，不只写“模型 A”。SongBloom 加 `sketch-head` 或 `acoustic-flow` 标签。

- 固定示例 `0 / 1 / 5 / 25 / 50 / 100 / 270`，左 A 永远同一 baseline，右 B 为当前 checkpoint；可用时显示原输入与 reference，但 reference 不是生成任务唯一 ground truth。
- 音频目标 60 秒，至少保留真实的 30 秒以上样本；不足则明示提前 EOS/时长，不造假补长。页面保留 waveform 与 mel，统一振幅、黑底与频率坐标。低于 48kHz 的原生音频不伪造 24kHz 以上真实带宽。
- 折叠曲线左 reward、右 KL；标清 `sampled token KL`、`conditional transition KL` 或 `window KL`，不能把不同单位的 KL 当同一纵轴。点上可查 step 与分数。
- valid / test 总表显示 `完成数/分母`，完整后才展示全量均值。主格为 `数值（Δ vs 同模型 baseline）`，缺失有原因；不拿 partial mean 当 final。
- 同时展示预先固定的普通示例和可追溯的单 reward badcase；后者列出完整集合分母、选择规则、checkpoint 与 score-to-audio SHA，标记事后选取，不伪装随机样本。
- 所有公开音频仅 **MP3，192 kbps / 48 kHz，不做响度归一化**；lossless 评分原件不进 repo，原件和 MP3 分别记录 SHA。盲听如用电平匹配副本，独立命名，保留原版，不能覆盖削波/响度证据。

## 9. 审批后执行顺序与工程验收

1. **先锁版本和输入。**收齐同事 10 条输入的 ID、具体 checkpoint/采样/时长/是否 reranking，以及逐条评分（如可提供）；缺失时明确报告，不把家族 MOS 绑定到新版本。完成 SongBloom 权重来源与参考音频的资源清单。
2. **DiffRhythm2 优先接入。**实现 native ODE + stochastic-policy baseline、transition log-prob 与 reference KL，先用自动化单元/数值验收检查符号、缓存、identity、梯度与 decoder 一致性。正式臂只有通过后才提交；验收不是一条“成功 RL”的 demo。
3. **SongBloom 并行准备。**建议 S1 先验证有限离散策略，S2 补声学主结论；版本未恢复或音频未验收则停在清楚的 blocked 状态，不换成 DiffRhythm1 冒名顶替。
4. **HeartMuLa 补中间档。**两层全部动作 likelihood 正确后再提交三个目标。YuE2 仅在批准匹配的新预算时建立相应新臂。
5. **集群按实际资源调度。**审批后再检查 Gadi wa66 / iv96、Kaya 与 5090；使用最高可用用户优先级但不抢占/杀别人的任务。新 flow / HeartMuLa 优先考虑 A100/H100 80GB；5090 资源满足时接入验证/单臂；Kaya V100 需独立 FP16 数值验收，不能把 dtype 差异当 LR 或模型质量效应。此处不是已提交的作业分配，也没有虚构排队 ETA。
6. **结束回收可复现物。**checkpoint 身份、训练步、optimizer/scheduler/RNG 状态、逐候选 SHA、完整 valid receipts、错误与续跑信息均核验后，才发布 MP3、曲线和 full-cohort 表格。失败臂也归档，不能只发布上涨的臂。

接入验收至少覆盖：零 adapter 与 frozen reference 身份一致；更新前 old/current ratio 近1并记录数值误差；有效组有有限非零梯度且冻结模块无更新；采样/重算在 CFG、mask、precision、batch 与缓存上逐项对齐；相同 latent 的 decoder 输出和 scored SHA 可复现。零 KL 的 identity test 不能替代一次非零扰动后的 KL 灵敏度检查。AR 改为无 top-k 等训练分布时，同样需要自己的 sampler-matched step0，不能直接移用同事 native 设置下的 MOS。

需要的 adapter 放在既有 `music-gen/lyrics2song/models/{heartmula,diffrhythm2,songbloom}/`，RL 适配放 `music-gen/lyrics2song/rl/grpo/`，本研究 manifest/results 留 `music-gen/reward-regime-study/`，展示仍在 `platform/site/demos/`。这些是未来落点，**本轮没有创建空壳 trainer 或自动运行脚本**。

## 10. 本轮核验记录与待确认项

| 项目 | 本次实际完成 | 尚未完成 |
| --- | --- | --- |
| HeartMuLa | 读取官方代码，核对 HF 元数据/许可；代码 SHA `5858ca8f1ffe58d62be7c007ea55f1033c5ca456` | 权重下载、GPU forward/backward、GRPO scorer |
| DiffRhythm2 | 读取作者发布分支代码/配置与官方论文；代码 SHA `7804f821b797b4f276090e1a9dcd37e97d9915d5` | 权重下载、SDE adapter、真实显存/吞吐与训练 |
| SongBloom | 官方论文/项目页仍可读；作者代码入口404、HF模型及long入口401；仅检查社区代码与文件元数据 | 作者/同事可核验的发布版本、权重 SHA、许可与音频资源、GPU运行 |
| 数据 | 重新核对现有 split 的270/30/121数量与 SHA；核对 WSB192、固定 audio prompt 和无音频发布的边界 | 对 SongBloom 所需参考资源的真实解码与覆盖审计 |
| 新实验 | 本 MD、已确认模型名单与demo独立实验选项 | **训练接入尚未完成、未排队、未训练，未更新为“已完成”demo** |

待落实的实现细节：**模型版本是否与同事快听一致；SongBloom S1先行还是直接S2；新矩阵60秒训练预算；SongBloom在WSB的固定参考输入另作标注。**模型名单已确认；其他默认沿用270/30、三个单reward、单LR3e-4和不启动定时任务。私有模型只占位，不运行。先完成用户要求的页面整理，不用未实现的trainer排空作业。

### 一手资料入口

- [HeartMuLa 官方代码、权重与许可](https://github.com/HeartMuLa/heartlib)；[技术报告](https://arxiv.org/abs/2601.10547)。
- [DiffRhythm2 论文](https://arxiv.org/abs/2510.22950)；[论文仓库](https://github.com/xiaomi-research/diffrhythm2)；[ASLP 发布分支](https://github.com/ASLP-lab/DiffRhythm2)；[发布权重](https://huggingface.co/ASLP-lab/DiffRhythm2)。
- [SongBloom 论文](https://arxiv.org/abs/2506.07634)；[作者项目页](https://cypress-yang.github.io/SongBloom_demo/)；[论文代码入口，当前404](https://github.com/Cypress-Yang/SongBloom)；[另一历史代码入口，当前404](https://github.com/tencent-ailab/SongBloom)；[作者权重入口，当前401](https://huggingface.co/CypressYang/SongBloom)。社区副本只作为明确标注的恢复线索，不能替代作者版本证明。
- [Flow-GRPO 论文及官方实现](https://github.com/yifan123/flow_grpo)。这里借鉴的是方法与算法实现，不声称它已经支持上述音乐模型。
- [SongBench 官方维度定义](https://github.com/Tencent/SongBench)；[SongEval 官方仓库](https://github.com/ASLP-lab/SongEval)。
- [CMI-Pref](https://huggingface.co/datasets/HaiwenXia/cmi-pref)；[WildSongBench](https://huggingface.co/datasets/m-a-p/WildSongBench)。私有占位不提供内部模型来源或实现信息。
