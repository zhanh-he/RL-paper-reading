
# Huawei RL 研究文献总目录

这个目录用于持续收集 Huawei RL / music generation post-training / reward design 方向的论文笔记。每篇文章单独成文，目录只保留导航、主题归类和对项目研发路线的启发。

维护约定：本窗口之后关于 Huawei RL、music RLHF、reward receipt、DPO/GSPO、music MOS/reward metric 的讨论，默认同步沉淀到这个 Obsidian vault。具体规则见 文献库维护约定。

## 文献索引（按 Zhanh 星级排序）

Felix 星级由 Felix 独立判断，当前刻意留空。年份与 venue 使用两行显示；`arXiv` 表示截至 2026-09-02 尚未核验到正式发表 venue。外部入口只保留已核验可访问的 `paper / github / demo / web`，没有公开入口时不编造链接。

| 年份 / Venue | Felix 星级 | Zhanh 星级 | 主题 | 文献全名 | 核心词汇 | 超链接 | 对项目的直接价值 |
|---|:---:|:---:|---|---|---|---|---|
| 2025<br>ICASSP |  | 5/5 | Accompaniment metric / perturbation validation / human study | Accompaniment Prompt Adherence: A Measure for Evaluating Music Accompaniment Systems | APA；reference-aware FAD；mismatched anchor；perturbation validation | [paper](https://arxiv.org/abs/2503.06346) · [github](https://github.com/SonyCSLParis/audio-metrics) · [web](https://pypi.org/project/audio-metrics/) | ICASSP metric qualification receipt：timing/pitch/nuisance perturbations + 875 次人工评分；但不是 per-song reward，也没有 policy-attack test。 |
| 2025<br>ISMIR |  | 5/5 | Stem accompaniment generation / beat F1 | STAGE: Stemmed Accompaniment Generation through Prefix-Based Conditioning | STAGE；prefix-based conditioning；context token；tempo-constrained generation；Beat This F1 | [paper](https://arxiv.org/abs/2504.05690) · [github](https://github.com/giorgioskij/stage) · [demo](https://giorgioskij.github.io/stage-demo/) · [web](https://ismir2025program.ismir.net/poster_269.html) | 已用 Beat This + `mir_eval` F1 评价 generated stem；证明 plain beat detection/F1 不是新贡献，必须增加 drift、meter ambiguity、robustness 和 human fit。 |
| 2026<br>Interspeech |  | 5/5 | Speech post-training / composite perceptual reward / GSPO | Post-Training Speech Enhancement Language Models with Perceptual Rewards | composite perceptual reward；DNSMOS；`1-WER`；UTMOS；GSPO | [paper](https://arxiv.org/abs/2606.21458) | `DNSMOS + (1-WER) + UTMOS` 的 composite reward 范式，并用 human evaluation 验证 composite 优于 single-metric variants；是 music reward receipt 的直接 audio 类比。 |
| 2026<br>AAAI |  | 5/5 | Speech restoration / unanimous multi-metric pairs / DPO | Multi-Metric Preference Alignment for Generative Speech Restoration | GenSR-Pref；unanimous multi-metric preference；Pareto dominance；DPO | [paper](https://ojs.aaai.org/index.php/AAAI/article/view/40775) · [demo](https://gensr-pref.github.io/) · [web](https://arxiv.org/abs/2508.17229) | 80K pairs 只接纳 perceptual quality、fidelity、content、timbre 一致判优的 winner；是 beat 改善且 coverage/quality 非劣 pair rule 的最直接 audio 先例。 |
| 2024<br>ICML |  | 5/5 | Music RLHF / sequence-level reward / human preference | MusicRL: Aligning Music Generation to Human Preferences | MusicRL；MusicLM-RL；reward model；RLHF；quality/adherence reward | [paper](https://proceedings.mlr.press/v235/cideron24a.html) · [demo](https://google-research.github.io/seanet/musiclm/rlhf/) | 证明 autoregressive music-token LM 可通过 sequence-level reward/RLHF 改善 text adherence 与 audio quality；支持音乐生成 post-training 的可行性。 |
| 2026<br>ICLR |  | 5/5 | Music reward hacking / policy-generated trajectories / adversarial post-training | Generative Adversarial Post-Training Mitigates Reward Hacking in Live Human-AI Music Interaction | GAPT；reward hacking；adversarial realism；co-evolving discriminator；policy trajectories | [paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/e6731a1d3eddafeb8c94669a1d82fc5d-Abstract-Conference.html) · [github](https://github.com/lukewys/realchords-pytorch) · [demo](https://realchords-gapt.github.io/) · [web](https://openreview.net/forum?id=FXm5U16vxD) | 音乐 reward hacking 的最直接公开证据：coherence reward 上升但简单和弦重复、diversity collapse，且 KL + rules 不足；要求检查真实 policy trajectories。 |
| 2026<br>ISMIR |  | 5/5 | Music GRPO / discrete diffusion / black-box audio rewards | DDSynth-RL: Audio Synthesizer Inversion via Discrete Diffusion with Reinforcement Learning | DDSynth-RL；discrete diffusion；black-box audio reward；GRPO；OOD retention | [paper](https://arxiv.org/abs/2608.03032) · [demo](https://ddsynth-rl.github.io/DDSynthRL-Demo/) | `K=8` rendered-audio GRPO 显著改善 OOD NSynth matching，却让 in-domain Dexed 退化；直接展示强推 reward 与 domain-retention trade-off。 |
| 2026<br>ICML |  | 5/5 | Music reward benchmark / multimodal preference | CMI-RewardBench: Evaluating Music Reward Models with Compositional Multimodal Instruction | compositional multimodal instruction；CMI-RewardBench；CMI-RM；reward decomposition | [paper](https://arxiv.org/abs/2603.00610) · [github](https://github.com/Haiwen-Xia/CMI-RewardBench) · [web](https://huggingface.co/datasets/HaiwenXia/cmi-pref) | 110K pseudo + 4,027 expert preferences、23 generators；是 holistic musicality/alignment 强 baseline，并留下 local temporal structure 与 head disentanglement 缺口。 |
| 2025<br>arXiv |  | 5/5 | Song aesthetics / professional full-song ratings / learned evaluator | SongEval: A Benchmark Dataset for Song Aesthetics Evaluation | SongEval；five-dimensional aesthetics；expert MOS；full-song evaluation | [paper](https://arxiv.org/abs/2505.10793) · [github](https://github.com/ASLP-lab/SongEval) · [demo](https://aslp-lab.github.io/SongEval/) · [web](https://huggingface.co/datasets/ASLP-lab/SongEval) | 2,399 full songs、140+ h、16 名专业标注者；适合 holistic guardrail/full-song stress test，但 Coherence 不能当 beat ground truth。 |
| 2025<br>ICASSP |  | 5/5 | Cross-generator music benchmark / human preference / metric meta-evaluation | Benchmarking Music Generation Models and Metrics via Human Preference Studies | AIME；cross-generator preference；metric meta-evaluation；same-prompt comparison | [paper](https://doi.org/10.1109/ICASSP49660.2025.10887745) · [web](https://huggingface.co/datasets/disco-eth/AIME) | 6,000 clips、12 systems、15,600 human pairs，含 Suno/Udio；适合公开 same-prompt disagreement mining 与 rhythm-focused relabel。 |
| 2026<br>arXiv |  | 5/5 | Long-form song reward / critique-conditioned scoring / GRPO | MuseCritic: Learning Multi-Aspect Song Rewards through Natural-Language Aesthetic Critiques | MuseCritic；critique-then-score；five-aspect rewards；Music Arena | [paper](https://arxiv.org/abs/2608.11755) · [github](https://github.com/WuqnEl/MuseCritic) | 五维 learned song reward；Music Arena 71.35% 是 overall preference 而非 beat。与 explicit BeatReward 互补，不能用 Musicality/Coherence 替代 rhythm-specific labels。 |
| 2026<br>ACL |  | 5/5 | Open long-form song generator / Qwen3-MuCodec / public GRPO backbone | Muse: Towards Reproducible Long-Form Song Generation with Fine-Grained Style Control | Muse；Qwen3；MuCodec；long-form song generation；MuseCritic-GRPO | [paper](https://arxiv.org/abs/2601.03973) · [github](https://github.com/yuhui1038/Muse) · [web](https://huggingface.co/bolshyC/models) | 公开 0.6B 模型、116K song data、训练/推理/评价 pipeline；是 public post-training backbone，但 full mix 需先过 frozen separator bridge。 |
| 2026<br>ICASSP |  | 5/5 | ICASSP challenge / generated-song aesthetics | The ICASSP 2026 Automatic Song Aesthetics Evaluation Challenge | ASAE；Top-Tier Accuracy；Hard set；unseen generator；fine-grained aesthetics | [paper](https://arxiv.org/abs/2601.07237) · [github](https://github.com/ASLP-lab/SongEval) · [demo](https://aslp-lab.github.io/Automatic-Song-Aesthetics-Evaluation-Challenge/) · [web](https://2026.ieeeicassp.org/sp-grand-challenges/) | 官方 venue signal：generated-song automatic evaluation 与 RL reward 有明确研究价值；Hard set 强调 unseen-generator generalization。 |
| 2024<br>arXiv |  | 5/5 | Preference noise / confidence filtering | Impact of Preference Noise on the Alignment Performance of Generative Language Models | preference noise；label flip；confidence filtering；downstream alignment | [paper](https://arxiv.org/abs/2404.09824) | 系统展示 preference noise 对 downstream alignment 的损害及 confidence filtering 的部分收益；直接支撑 pair correctness 与 noise-curve 实验。 |
| 2026<br>arXiv |  | 5/5 | Selective GRPO / NLL anchor | Adaptive Loss Balancing for Noise-Robust GRPO in Generative Recommendation | AdaGRPO；selective admission；NLL fallback；policy difficulty；reward discriminability | [paper](https://arxiv.org/abs/2606.08480) | 只在 policy difficulty 和 reward discriminability 均通过时使用 GRPO，否则回到监督 NLL；是 confidence-gated GRPO 的直接方法先例。 |
| 2022<br>NeurIPS |  | 5/5 | Reward hacking / proxy impossibility | Defining and Characterizing Reward Hacking | hackability；unhackable proxy；policy-order equivalence；Goodhart | [paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/3d719fee332caa23d5038b8a90e81796-Abstract-Conference.html) | 理论上说明全策略空间的非平凡 unhackable proxy 需要极强的 policy-order equivalence；支持限制 policy set，而非声称完美 reward。 |
| 2024<br>ICLR |  | 5/5 | Composite RM / constrained RLHF | Confronting Reward Model Overoptimization with Constrained RLHF | constrained RLHF；usefulness threshold；dynamic Lagrange；composite reward model | [paper](https://proceedings.iclr.cc/paper_files/paper/2024/hash/5eee634cb9729b8bcc2ec9f2a46a74ae-Abstract-Conference.html) | 固定多 reward 加权存在不同 usefulness turning points；支持 beat objective + coverage/quality constraints，而非无限补偿式求和。 |
| 2024<br>NeurIPS |  | 5/5 | DPO / direct alignment overoptimization | Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms | direct-alignment overoptimization；KL budget；DPO scaling law；early degradation | [paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/e45caa3d5273d105b8d045e748636957-Abstract-Conference.html) | DPO 也会随 KL budget 增大而退化，甚至一个 epoch 内发生；DPO 更可审计，但不免疫 overoptimization。 |
| 2025<br>CoLM |  | 5/5 | Policy-induced shift / reward refresh | Off-Policy Corrected Reward Modeling for Reinforcement Learning from Human Feedback | off-policy correction；current-policy data；distribution mismatch；reward refresh | [paper](https://openreview.net/forum?id=0zxugBcgF5) · [web](https://arxiv.org/abs/2507.15507) | 将 overoptimization 解释为 reward-data 与 current-policy rollout mismatch；要求逐 checkpoint 重测 reward validity。 |
| 2026<br>ACL |  | 4.5/5 | Multimodal generative reward benchmark | Omni-RewardBench: Toward a Comprehensive Evaluation of Generative Reward Models Across Modalities | Omni-RewardBench；triplet evaluation；modality dominance；perception/fusion failure | [paper](https://aclanthology.org/2026.acl-long.636/) | 揭示 perception failure、modality dominance、fusion failure；支持独立 MIR verifier 作为 learned judge 的感知审计层。 |
| 2026<br>arXiv |  | 4.5/5 | Speech enhancement / online GRPO / reward hacking | FlowSE-GRPO: Training Flow Matching Speech Enhancement via Online Reinforcement Learning | FlowSE-GRPO；single-metric hacking；DNSMOS；speaker similarity；multi-metric reward | [paper](https://arxiv.org/abs/2601.16483) | 单独优化 DNSMOS、speaker similarity 或 SpeechBERTScore 会快速提分但损害其他 fidelity；证明 GRPO 能强推 proxy，真正风险是 policy-induced hacking。 |
| 2026<br>CVPR |  | 4.5/5 | Flow-GRPO / implicit overoptimization / regulated clipping | GRPO-Guard: Mitigating Implicit Over-Optimization in Flow Matching via Regulated Clipping | GRPO-Guard；implicit overoptimization；regulated clipping；flow importance ratio | [paper](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_GRPO-Guard_Mitigating_Implicit_Over-Optimization_in_Flow_Matching_via_Regulated_Clipping_CVPR_2026_paper.html) · [github](https://github.com/yifan123/flow_grpo) · [demo](https://jingw193.github.io/GRPO-Guard/) | 揭示 proxy 上升、gold quality 下降的 optimizer-side 机制；即使底座不是 flow，也必须迁移 proxy/gold trajectory 与 clipping/ratio audit。 |
| 2026<br>ISMIR |  | 4.5/5 | Reward shortcut / genre bias / SongEval audit | Genre Bias or Aesthetic Perception? Identifying and Mitigating Shortcut Learning in Music Evaluation | shortcut learning；genre bias；SongEval audit；counterfactual augmentation | [paper](https://arxiv.org/abs/2607.13903) | 证明 aesthetics predictor 可能把 genre 当 quality proxy；要求对 MIR reward 做 tempo/density/confidence subgroup audit，并降低 aesthetics correlation 的证据等级。 |
| 2024<br>ISMIR |  | 4.5/5 | Public beat/downbeat detector | Beat This! Accurate Beat Tracking Without DBN Postprocessing | Beat This；DBN-free tracking；beat/downbeat；focal loss；transformer | [paper](https://arxiv.org/abs/2407.21658) · [github](https://github.com/CPJKU/beat_this) | 公开、可复现的 detector transfer baseline；用于替代内部 Beat It，但 detector 本身不是 BeatReward novelty。 |
| 2026<br>arXiv |  | 4.5/5 | Multi-reward GRPO / normalization | GDPO: Group Reward-Decoupled Normalization Policy Optimization for Multi-Reward RL Optimization | GDPO；reward-decoupled normalization；multi-reward GRPO；variance hijacking | [paper](https://arxiv.org/abs/2601.05242) | 指出先求和再 group z-score 会让 reward 组合坍缩；beat + quality + coverage 应逐维归一化或采用 gate/Pareto。 |
| 2025<br>arXiv |  | 4.5/5 | Music metric / human preference meta-evaluation | Aligning Text-to-Music Evaluation with Human Preferences | text-to-music evaluation；human preference alignment；metric meta-evaluation；pairwise validity | [paper](https://arxiv.org/abs/2503.16669) | 常见 TTM 指标与 human preference 可能很弱；支持 controlled construct test + real human validation + pair-level evaluation。 |
| 2025<br>NeurIPS |  | 4.5/5 | Song generation / multi-preference DPO / Chinese traditional instruments | LeVo: High-Quality Song Generation with Multi-Preference Alignment | LeVo；mixed/dual-track tokens；multi-preference DPO；semi-automatic pairs | [paper](https://arxiv.org/abs/2506.07520) · [demo](https://levo-demo.github.io/) | 最接近 song-level multi-preference DPO；可借鉴多维偏好拆分与 dual-track tokens；官方 GitHub 于 2026-09-02 返回 404，故暂不列死链接。 |
| 2025<br>arXiv |  | 4.5/5 | Lyric-to-song / noisy PER / DPO-PPO-GRPO comparison | Towards Hallucination-Free Music: A Reinforcement Learning Framework for Hallucination Control | hallucination-free music；phoneme error rate；DPO/PPO/GRPO；lyric alignment | [paper](https://arxiv.org/abs/2508.05011) | same-prompt PER filtering 与 DPO/PPO/GRPO 对照；objective reward 和 subjective lyric consistency 未完全同向，直接支撑 human guardrail。 |
| 2023<br>ICML |  | 4.5/5 | Proxy reward overoptimization / Best-of-N / RL | Scaling Laws for Reward Model Overoptimization | overoptimization scaling law；proxy/gold reward；Best-of-N；KL distance | [paper](https://proceedings.mlr.press/v202/gao23h.html) | Best-of-N 与 RL 都可能随优化压力增加而偏离 gold；要求 Best-of-K 同时报告 proxy、control 与 human 曲线。 |
| 2024<br>ICLR |  | 4.5/5 | Reward ensembles / conservative optimization | Reward Model Ensembles Help Mitigate Overoptimization | worst-case optimization；uncertainty weighting；reward ensemble；conservative RLHF | [paper](https://proceedings.iclr.cc/paper_files/paper/2024/hash/dda7f9378a210c25e470e19304cce85d-Abstract-Conference.html) | worst-case/uncertainty-weighted ensembles 配合小 KL 可缓解 overoptimization；共享 blind spot 仍需攻击测试。 |
| 2024<br>NeurIPS |  | 4.5/5 | DPO + SFT anchor / robust regularization | Provably Mitigating Overoptimization in RLHF: Your SFT Loss Is Implicitly an Adversarial Regularizer | RPO；SFT anchor；adversarial regularizer；partial coverage | [paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/fa69e968b7319fd42524febd41475fb3-Abstract-Conference.html) | 从 distribution shift/uncertainty 推导 DPO + SFT anchor；支持 pair filtering 之外约束 reference drift。 |
| 2025<br>IJCAI |  | 4.5/5 | Symbolic music / CLaMP-DPO / automatic preference construction | NotaGen: Advancing Musicality in Symbolic Music Generation with Large Language Model Training Paradigms | CLaMP-DPO；DPOP；reference anchor；iterative DPO；K iterations | [paper](https://arxiv.org/abs/2502.18008) · [github](https://github.com/ElectricAlexis/NotaGen) · [demo](https://electricalexis.github.io/notagen-demo/) | 自动 top/bottom preference construction、K=0 vs K=3、DPOP 与 human A/B 是我们的 pair recipe 和 iterative DPO 强参考。 |
| 2025<br>arXiv |  | 4.5/5 | GSPO / online sequence-level RL / Qwen | Group Sequence Policy Optimization | GSPO；sequence-level importance ratio；group sequence；sequence reward | [paper](https://arxiv.org/abs/2507.18071) | sequence-level ratio 更贴合整段音乐 reward；适合 reward 成熟后的 P2，不替代当前 pair/reward qualification。 |
| 2019<br>ICML |  | 4.5/5 | Selective prediction / abstention / AURC | SelectiveNet: A Deep Neural Network with an Integrated Reject Option | SelectiveNet；reject option；risk-coverage；selective prediction；AURC | [paper](https://proceedings.mlr.press/v97/geifman19a.html) | 决定 confidence 必须同时报告 accepted accuracy 与 coverage/AURC，而不是只报过滤后 accuracy。 |
| 2025<br>arXiv |  | 4.5/5 | Sample-wise noisy preference / EM reweighting | RE-PO: Robust Enhanced Policy Optimization as a General Framework for LLM Alignment | RE-PO；correctness posterior；EM reweighting；sample-wise noise | [paper](https://arxiv.org/abs/2509.24159) | 推断每条 label 的 posterior correctness 并重加权 preference loss；形式上最接近 confidence-weighted MIR pairs。 |
| 2026<br>ICLR |  | 4.5/5 | Multi-dimensional preference conflict / Semi-DPO | Learning from Noisy Preferences: A Semi-Supervised Learning Approach to Direct Preference Optimization | Semi-DPO；multidimensional conflict；unlabeled pairs；semi-supervised preference | [paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/17061a94c3c7fda5fa24bbdd1832fa99-Abstract-Conference.html) | 多维冲突被压成 binary label 会制造错误梯度；支持 consensus/Pareto clean pairs，冲突 pair abstain。 |
| 2024<br>arXiv |  | 4.5/5 | Reward uncertainty / LoRA ensemble | Uncertainty-Penalized RLHF with Diverse Reward LoRA Ensembles | UP-RLHF；reward LoRA ensemble；epistemic uncertainty；uncertainty penalty | [paper](https://arxiv.org/abs/2401.00243) | 支持用异构 reward ensemble 估计不确定性并保守优化；tracker 投票仍不等于正确。 |
| 2024<br>CoLM |  | 4.5/5 | Ensemble reward hacking / shared blind spots | Helping or Herding? Reward Model Ensembles Mitigate but Do Not Eliminate Reward Hacking | Helping or Herding；shared blind spots；ensemble hacking；diversity | [paper](https://arxiv.org/abs/2312.09244) | ensemble 可缓解却不能消除 reward hacking；Madmom + Beat This 只能提供 uncertainty evidence。 |
| 2025<br>ACL |  | 4.5/5 | Fine-grained audio evaluator / human validation | T2A-Feedback: Improving Basic Capabilities of Text-to-Audio Generation via Fine-Grained AI Feedback | T2A-Feedback；event occurrence；event sequence；fine-grained AI feedback | [paper](https://arxiv.org/abs/2505.10561) | evaluator 先验证 human preference，再构造 249K audio preference data；强支撑 evaluator-first protocol。 |
| 2023<br>NeurIPS |  | 4/5 | DPO / offline preference optimization | Direct Preference Optimization: Your Language Model Is Secretly a Reward Model | DPO；implicit reward model；Bradley-Terry；reference policy | [paper](https://arxiv.org/abs/2305.18290) | 当前 DPO 数学基础；关键贡献不在公式，而在 chosen/rejected 是否真实代表更适合 vocal 的伴奏。 |
| 2026<br>arXiv |  | 4/5 | Video RLVR / verifiable rewards / dense reward design | Video Models Can Reason with Verifiable Rewards | VideoRLVR；verifiable reward；dense decomposition；SDE-GRPO；KL guard | [paper](https://arxiv.org/abs/2605.15458) · [github](https://github.com/luka-group/VideoRLVR) · [demo](https://darthzhu.github.io/VideoRLVR-page/) | 证明生成模型可用 rule-based verifiers + dense decomposed rewards 训练；可迁移为 hard checks + MIR + learned perceptual + human audit。 |
| 2025<br>NeurIPS |  | 4/5 | Flow matching / online GRPO / ODE-to-SDE | Flow-GRPO: Training Flow Matching Models via Online RL | Flow-GRPO；online RL；ODE-to-SDE；denoising reduction | [paper](https://arxiv.org/abs/2505.05470) · [github](https://github.com/yifan123/flow_grpo) · [demo](https://gongyeliu.github.io/Flow-GRPO/) | 建立 flow-model online RL 基础；对当前 AR Qwen 主要是算法边界和 rollout engineering 参考。 |
| 2025<br>arXiv |  | 4/5 | Music-generation evaluation taxonomy / metric validity | A Survey on Evaluation Metrics for Music Generation | metric taxonomy；objective evaluation；human evaluation；cultural bias；standardization | [paper](https://arxiv.org/abs/2509.00051) | 系统整理 symbolic/audio metrics 与 human evaluation 缺口；用于 related-work taxonomy，不作为单一方法证据。 |
| 2026<br>Web |  | 4/5 | RLVR literature review / verifiable reward codebase registry | Awesome RLVR Literature Review and Resource Registry | RLVR；verifiable reward；survey registry；codebase map；reward receipt | [github](https://github.com/opendilab/awesome-RLVR) · [web](https://rlvrbook.com/) | 把 music reward receipt 放入 hard checks + semi-verifiable MIR + learned rewards + human validation 的统一 RLVR 图谱。 |
| 2026<br>arXiv |  | 4/5 | Layer-selective RL / Qwen / LoRA placement | Is One Layer Enough? Training a Single Transformer Layer Can Match Full-Parameter RL Training | single-layer RL；layer contribution；middle-layer tuning；LoRA placement | [paper](https://arxiv.org/abs/2607.01232) | 提醒不要默认全参或全层 LoRA；应先做 middle-layer/contribution-guided LoRA DPO 小规模验证。 |
| 2026<br>arXiv |  | 4/5 | Anchored ranking / listwise DPO | From Probability to Advantage: Unifying Alignment via Anchored Ranking | ADPO；anchored ranking；reference-relative advantage；ordinal/listwise preference | [paper](https://arxiv.org/abs/2510.18913) | 可利用同一 vocal 的全部 K candidates 做 ordinal/listwise alignment；当前缺少 audio validation，只作为 P1 stretch。 |
| 2026<br>arXiv |  | 4/5 | GRPO + DPO + UNA / online advantage matching | GIFT: Group-Relative Implicit Fine-Tuning Integrates GRPO with DPO and UNA | GIFT；group-relative implicit reward；UNA；explicit/implicit advantage matching | [paper](https://arxiv.org/abs/2510.23868) | 匹配 explicit reward 与 DPO implicit reward 的 group-relative advantages；适合 reward 成熟后的 online exploration。 |
| 2026<br>ISMIR |  | 4/5 | Multi-stem aesthetics / uncertainty | Song Aesthetics Evaluation with Multi-Stem Attention and Hierarchical Uncertainty Modeling | multi-stem attention；hierarchical uncertainty；vocal-accompaniment relation；interval prediction | [paper](https://arxiv.org/abs/2601.12222) · [github](https://github.com/yisan33/song-aesthetics-evaluation) | 支持 stem decomposition 与 uncertainty；也迫使我们突出无需 MOS 的 explicit timing construct 和 confidence calibration。 |
| 2024<br>arXiv |  | 4/5 | Random-flip robust DPO | Provably Robust DPO: Aligning Language Models with Noisy Feedback | robust DPO；symmetric label noise；noise correction；preference flip | [paper](https://arxiv.org/abs/2403.00409) | 适合 random-noise baseline，但不覆盖 sample-dependent、结构化 MIR detector error。 |
| 2025<br>arXiv |  | 4/5 | Noise-corrected GRPO / unbiased gradient | Noise-Corrected GRPO: From Noisy Rewards to Unbiased Gradients | noise-corrected GRPO；Bernoulli corruption；flip probability；unbiased gradient | [paper](https://arxiv.org/abs/2510.18924) | 对 Bernoulli reward corruption 修正 gradient；对连续、相关、结构化音乐噪声只作数学对照。 |
| 2024<br>NeurIPS |  | 4/5 | Soft preference labels / DPO | Geometric-Averaged Preference Optimization for Soft Preference Labels | geometric averaging；soft preference；probabilistic label；tie-aware DPO | [paper](https://arxiv.org/abs/2409.06691) | 接近 50/50 的 pair 学习压力趋零，适合 calibrated confidence-weighted music DPO。 |
| 2022<br>arXiv |  | 4/5 | Conformal risk control | Conformal Risk Control | conformal risk control；risk-coverage；finite-sample guarantee；calibration set | [paper](https://arxiv.org/abs/2208.02814) · [github](https://github.com/aangelopoulos/conformal-risk) | private calibration labels 足够时可控制 accepted-pair risk；当前先做 frozen threshold、AURC 与 cluster bootstrap。 |
| 2026<br>arXiv |  | 4/5 | Text-to-audio / LALM reward / online GRPO | Resonate: Reinforcing Text-to-Audio Generation via Online Feedback from Large Audio Language Models | Resonate；LALM feedback；online GRPO；text-to-audio reward | [paper](https://arxiv.org/abs/2603.11661) | 说明 reward 成熟时 online GRPO 可优于 offline；不支持“DPO 普遍优于 GRPO”，反而强调先修 reward。 |
| 2026<br>arXiv |  | 4/5 | Music quality MOS-like metric | MuQ-Eval: An Open-Source Per-Sample Quality Metric for AI Music Generation Evaluation | MuQ-Eval；per-sample quality；frozen MuQ；MusicEval；MOS correlation | [paper](https://arxiv.org/abs/2603.22677) · [github](https://github.com/dgtql/MuQ-Eval) | 最像 music DNSMOS/UTMOS 的开放 per-sample quality predictor；可用于 Best-of-K 和 DPO quality gate，但对结构扰动不敏感。 |
| 2026<br>arXiv |  | 4/5 | Text-to-music pairwise reward model | TuneJury: An Open Metric for Improving Music Generation Preference Alignment | TuneJury；pairwise reward；anchor calibration；Bradley-Terry；expert iteration | [paper](https://arxiv.org/abs/2606.17006) · [github](https://github.com/yonghyunk1m/TuneJury) · [demo](https://huggingface.co/spaces/TuneJury/tune-jury-demo) · [web](https://huggingface.co/spaces/TuneJury/tune-jury) | 开放 instance-level pairwise reward；可做 Best-of-N、pair filtering 和 OOD anchor calibration，但需验证 vocal-fit/beat sensitivity。 |
| 2023<br>OUP |  | 3.5/5 | Neuroscience of music reward | Music Recruits the Reward System | music reward system；hedonic valuation；prediction；mesolimbic response | [web](https://academic.oup.com/book/55154/chapter/424072391) | 支持区分 structural perception 与 hedonic valuation；不能用来声称 beat alignment 等价于 pleasure。 |
| 2026<br>ICML |  | 3.5/5 | Unsupervised source separation / remixing flow / domain adaptation | SURF: Separation via Unsupervised Remixing Flow | SURF；unsupervised remixing；teacher-student flow；source separation | [paper](https://arxiv.org/abs/2606.04921) · [demo](https://google.github.io/df-conformer/surf/) | 用于分析 separator-induced BeatReward noise、构造 remix stress tests；不设为截稿前复现依赖。 |
| 2024<br>arXiv |  | 3.5/5 | Music-understanding foundation-model survey | A Survey of Foundation Models for Music Understanding | music foundation models；pretrained representation；audio language model；survey taxonomy | [paper](https://arxiv.org/abs/2409.09601) | 支持 learned semantic judge + explicit MIR expert 分层；模型地图停留在 2024，不代表当前 SOTA。 |
| 2026<br>ICASSP |  | 4/5 | Flow-DPO / multi-reward preference optimization | MR-FlowDPO: Multi-Reward Direct Preference Optimization for Flow-Matching Text-to-Music Generation | MR-FlowDPO；MRSD；strong-dominating pairs；flow matching；multi-reward DPO | [paper](https://arxiv.org/abs/2512.10264) · [demo](https://lonzi.github.io/mr_flowdpo_demopage/) | CLAP、production quality、music-HuBERT semantic consistency 三轴构造 strong-dominating pairs；迁移 pair recipe，不直接套 flow loss。 |
| 2023<br>arXiv |  | 3/5 | Vocal-to-accompaniment / leakage | SingSong: Generating Musical Accompaniments from Singing | AudioLM adaptation；source-separated pairs；vocal conditioning；leakage；noise augmentation | [paper](https://arxiv.org/abs/2301.12662) · [github (unofficial)](https://github.com/jihoojung0106/open-singsong) · [demo](https://storage.googleapis.com/sing-song/index.html) | 支撑 separated-vocal leakage 和 noise/artifact augmentation；preference receipt 必须记录 separator 与 residual leakage。 |
| 2026<br>arXiv |  | 3/5 | Accompaniment codec-token generation | HAFM: Hierarchical Autoregressive Foundation Model for Music Accompaniment Generation | HAFM；dual-rate tokenization；HuBERT；EnCodec；three-stage autoregression | [paper](https://arxiv.org/abs/2604.09054) · [github](https://github.com/HackerHyper/HAFM) · [web](https://huggingface.co/zhuqijian/HAFM) | 与 Qwen codec-token accompaniment generation 形态接近；dual-rate semantic/acoustic representation 可启发分层 reward。 |
| 2026<br>arXiv |  | 3/5 | Masked diffusion / nominal-only anomaly detection | Masked Diffusion Modeling for Anomaly Detection | MaskDiff-AD；masked diffusion；reconstruction surprisal；nominal-only anomaly detection | [paper](https://arxiv.org/abs/2605.30046) · [github](https://github.com/lxzhang1/MaskDiff-AD) | 可作 reward-hacked output OOD guardrail/diagnostic；原实验不是音频，近期不作主 reward。 |
| 2020<br>Web |  | 2/5 | AR vs NAR 中文入门 | 自回归（Autoregressive, AR）模型与非自回归（Non-Autoregressive, NAR）模型 | autoregressive；non-autoregressive；teacher forcing；parallel decoding | [web](https://www.cnblogs.com/ytxwzqin/p/12813965.html) | 便于团队沟通生成依赖和速度；概念混合经典时间序列 AR 与 neural autoregression，不作为论文证据。 |

## 星级说明

- Felix 星级：由 Felix 独立判断，当前全部留空。
- Zhanh 星级：表示对 Huawei RL 当前 research output 的直接相关性，不等于论文整体质量评分。
- 5/5：战略核心文献，直接支撑 research output 或方法路线。
- 4.5/5：高度相关的实现型文献或强 baseline，足以进入近期技术路线讨论。
- 4/5：当前项目重点文献，能影响 reward、pair construction、training 或 evaluation。
- 3.5/5：有明确技术启发，但主要作为模块或对照。
- 3/5：背景参考，遇到对应问题时回看。

## 当前主题地图

- Useful Literature: Dynamic Policy Data and Reward Validity：本轮 literature review 的忠实归档；按项目相关性分级汇总 policy-induced distribution shift、reward overoptimization、multi-reward conflict、confidence/DPO/GRPO 与 music/audio 证据，并给出 ICASSP citation jobs 和八步实验主线。
- Offline accuracy != online validity：领导汇报版理论主线；纠正 scalar-invariant 术语，把解法扩展为 confidence/Pareto DPO、robust reward system、conservative GRPO 与 current-policy refresh 四层防线。
- GAPT music reward hacking：当前最直接的公开音乐证据；coherence reward 高但策略坍缩为重复简单和弦，说明 artificial corruption accuracy 和 KL/rule guardrail 都不足，必须检查 policy-generated trajectories。
- GRPO overoptimization -> Pareto DPO 主线：当前总判断；代码审计确认 v5 GRPO beat-only，提出 adversarial scorer tests、checkpoint trajectory、public SongGen/CMI-Pref 和 confidence/Pareto DPO。
- Experiment Design v7：当前实验版本；v5 scorer 与 v6 confidence 保留，优化路线改为 offline-online mismatch + Pareto DPO。
- Confidence-aware / noisy-reward 主线：把 confidence 定义为 pair reward direction 的 correctness probability，用 risk-coverage/AURC 评价，并将 reranking、DPO、GRPO 变成 selective supervision；当前优先级高于继续做固定 composite score。
- BeatReward v6：confidence/risk-coverage canonical provenance；已被 v7 实验路线 supersede，但定义和结果仍有效。
- v6 public result：v2/v5 raw vs non-tie ranking、risk-coverage 与 metrical-alias high-confidence failure。
- ICASSP / BeatReward 主线：把论文收敛为 confidence-aware、perturbation-validated vocal-accompaniment beat reward；metric/meta-evaluation 是核心，reranking/DPO 是 downstream utility。
- ICASSP 2026 ASAE Challenge：最强 venue signal；其 unseen-generator Hard set、structure-aware winner 与 top-tier bottleneck直接决定我们的 evaluation framing。
- MuseCritic：公开五维 critique-conditioned song reward 与 GRPO baseline；本地 controlled audit 证明它能识别 clean/shifted，却不能稳定区分 70/120 ms severity，不能把 Coherence 当 beat。
- BeatReward v4：historical onset-grid/ensemble 设计与 signed-offset、metrical-alias provenance。
- BeatReward v3：保留 strict + onset universal fusion 的 evaluated negative-result provenance。
- Beat This：公开 detector transfer；论文贡献必须位于 detector 之后的 pairwise reward construct 与 meta-evaluation。
- Genre-bias audit：SongEval aesthetics 可能有 shortcut；不允许用其五维分数调 BeatReward。
- Multi-stem aesthetics：stem interaction 与 uncertainty 已是 concurrent direction；我们的区分点是 explicit timing、controlled corruption、locality 与 abstention。
- APA：最近的 ICASSP accompaniment metric 范式；其 perturbation + human validation 应直接进入我们的实验设计。
- STAGE：plain beat F1 的最近工作；决定 BeatReward 的 novelty 最低门槛。
- Omni-RewardBench：通用 multimodal judge 可能在底层感知、模态平衡和跨模态融合上失败，独立 MIR verifier 可提供审计证据。
- GDPO / ADPO / GIFT：分别回答 multi-reward normalization、offline listwise ranking、online explicit-implicit advantage matching；当前依次作为 P2 guardrail、P1 stretch、P2 stretch。
- Composite reward for post-training：speech paper 的关键不是 speech，而是把 perceptual quality、content preservation、naturalness 组合成 reward，并用 human evaluation 防 reward hacking。
- FlowSE-GRPO：single audio metric 在 online GRPO 中会快速提分并 reward hack；证明强 proxy 需要独立 guardrails 和 optimization-path audit。
- DDSynth-RL：公开音乐 GRPO + 多 audio reward + 可听 demo；OOD matching 大幅改善但 in-domain 退化，是 policy-shift/retention trade-off 的核心案例。
- GRPO-Guard：proxy reward 上升而 gold quality 下降的 optimizer-side 机制与修复；要求我们补 ratio/clip/gradient statistics。
- Flow-GRPO：flow matching online RL 的基础工程；与 GRPO-Guard 共 repo 但 demo page 和研究问题不同。
- GenSR-Pref：multi-metric unanimous winner + DPO；支持把 beat 当 objective、coverage/noise/quality 当 non-inferiority constraints。
- Music RLHF：音乐生成可以被 human preference / reward post-training 改善，但训练成本和标注成本高。
- Song multi-preference DPO：song generation 里可以把 lyrics、prompt、musicality、instrument/style 拆成多偏好；对我们是 multi-reward DPO 的直接参考。
- Hallucination-free song generation：DPO/PPO/GRPO 对比可作为我们方法排序的外部证据。
- MR-FlowDPO：flow-matching T2M 的 multi-reward DPO；MRSD 强支配 pair、数据驱动 margin 与独立 human/control evaluation 可直接进入我们的 DPO baseline。
- AIME human-preference benchmark：ICASSP 2025 的 12-generator、15,600-pair 公共基准；原 labels 作 overall guardrail，新增 rhythm-focused labels 作 BeatReward truth。
- SongEval paper：full-song professional aesthetics 与可听高低分 demo；必须保持 Coherence/Structure/Musicality 与 beat correctness 的构念边界。
- Reward overoptimization：Best-of-K 与 RL 都可能过度优化 imperfect proxy；决定我们必须 sweep K/steps 并保留独立 human/control evaluation。
- 14_Same-Prompt_Pairs_Risk-Coverage_and_Best-of-K：把 same-prompt ranking、selective risk 与 inference-time utility 串成 post-training 前的三步 go/no-go。
- CLaMP-DPO / automatic preference construction：每个 prompt 用 ground-truth embedding anchor 给 candidates 排序，top/bottom quantile 构造 DPO pairs；对我们 reward receipt 很有启发。
- GSPO vs DPO：GSPO 适合后续直接优化 composite reward，但当前第一版应先保留 DPO，因为 pair 可解释、可回放、可 ablate。
- VideoRLVR / verifiable rewards：可验证 reward 不应只做 sparse success，而要拆成 dense decomposed rewards；音乐里可迁移为 hard checks + MIR diagnostics + learned perceptual score + human A/B。
- awesome-RLVR：把 RLVR 作为大领域综述入口；对我们是 reward receipt、verifier-guided reranking、DPO pair construction 和后续 GSPO/GRPO 的统一路线图。
- Is One Layer Enough?：RL post-training 的收益可能集中在中层；对 Qwen-DPO/LoRA 的直接问题是 full-parameter、all-layer LoRA、middle-layer LoRA 哪个更稳。
- Music MOS / reward metrics：音乐里没有单一 UTMOS/DNSMOS 等价物，应该拆成 quality、alignment、naturalness/aesthetics 和 guardrails。
- Vocal-conditioned accompaniment data strategy：separated vocal leakage、train/inference mismatch、noise/artifact augmentation、decoding config 都要写入 pair metadata。

## 方法路线索引

| route | status | key notes |
|---|---|---|
| GRPO overoptimization audit | P0 / current | base/25/50/75/100/150 proxy-control-human trajectory；先判断 Goodhart 与 early-stop point |
| Pareto/unanimous DPO | P0 / current | beat margin/confidence通过且 coverage/noise/quality non-inferior；冲突 pair abstain |
| reward-direction confidence | P0 / current | development fit、validation threshold、test AURC；先验证能否识别错误 pair |
| selective Best-of-K | P0 / current | 只在高置信 pair/group 上 rerank；最先验证 human utility |
| best-of-K reranking | P0 | 用于验证 reward/scorer，不改模型 |
| offline DPO | P0/P1 | 只在 chosen/rejected direction 通过 confidence/human calibration 后进入 |
| soft / confidence-weighted DPO | P1 | 用 calibrated probability 表示 ambiguous preference；对照 hard filter |
| Anchored ADPO | P1 stretch | 对同一 vocal 的 K candidates 做 ordinal/listwise preference learning；先有 DPO baseline |
| multi-reward DPO | broad v1 / P1 | 长期 reward-recipe 路线；当前 v4 论文只把 BeatReward-DPO 当 downstream utility |
| DPO variants | P1/P2 | TDPO / SimPO / KTO 用于 length、reference cost、pair scarcity 问题 |
| selective GRPO + SFT anchor | P1/P2 stretch | reward direction、group logging、coverage 过 gate 后小规模做；不可靠 group 回到 SFT/NLL |
| GSPO / GRPO / GDPO | P2 | reward 稳定、online rollout ready 后再做；必须显式比较 `scale_rewards` 与 raw group spread |
| GIFT | P2 stretch | online group rollout + explicit/implicit advantage matching；不作为 ICASSP 截稿依赖 |
| PPO / reward model | P3 | 老式 RLHF baseline，当前不优先 |

## 数据集与资源索引

| 类型 | 资源 | 状态 | 对项目的直接价值 |
|---|---|---|---|
| Public validation map | Public Datasets for MIR Reward Validation | 已整理 / 当前入口 | 区分 stem construct validation、generated-song stress test 与 internal utility；包含 MoisesDB、MUSDB18、Slakh2100、CMI-Pref、SongEval、Muse。 |
| Same-prompt cross-generator human benchmark | AIME | 公开 / payload 待本地精确审计 | 6,000 generated tracks、12 systems、15,600 human pairs；含 Suno/Udio，适合公开 disagreement mining 和 rhythm relabel。 |
| Public trainable backbone | SongGen ICML 2025 dual-track | training code/checkpoints released / terms待逐项核对 | 1.3B AR、30s、vocal/accompaniment 分轨；当前 public Best-of-K/DPO 首选。 |
| Large cross-generator pool | CMI-Pref-Pseudo | 56K generations / 23 models / 165K pairs | OOD/disagreement mining；pseudo labels 不能作为 beat GT，需 rhythm-specific relabel。 |
| Aligned stems | MoisesDB / MUSDB18 | 已调研 / license gate | 从自然对齐 stems 构造已知 timing perturbations，是 BeatReward 可验证 ground truth 的主体。 |
| Exact timing supplement | Slakh2100 | 已调研 / CC BY 4.0 | aligned MIDI + stems，适合 meter、tempo、offset、drift 组件验证；无歌声。 |
| Generated-song stress test | SongEval / ASAE Challenge | payload 已精确审计 / NC | 2,400 audio、2,399 metadata，缺 ID 315；metadata 无 generator/genre/prompt。适合 frozen external stress test，不是 stem truth。 |
| Large SunoV5 corpus | Muse | 已整理 / payload blocked | 论文称 116k 首、约 7,771h，但当前 HF 数据页为空且仅约 2.67 kB；适合未来 long-form stress test，不能进入当前关键路径。 |
| Song generation code / weights | SongGeneration | 已收录 / license 风险 | 可看 design，不直接商用复用。 |
| Generated music MOS data | MusicEval | 待整理 | 可训练/验证 generated music quality 和 text alignment metric。 |
| Music quality metric | MuQ-Eval | 待整理 | P0 music quality scorer 候选。 |
| Pairwise reward model | TuneJury | 待整理 | P0/P1 preference score 候选。 |
| Multimodal reward model | CMI-RM | 已整理 / learned baseline | 未来 vocal/style/text/audio prompt reward 候选，也是当前 BeatReward complementarity baseline。 |
| Traditional instrument datasets | CCMusic / ChMusic / FolkMusic / Guzheng Tech datasets | 已在 LeVo KB 记录 | 多数有 NC/ND 或边界不清；先作研究评测和 benchmark。 |

## 后续收录模板

每来一篇新文章，用同一套结构：

1. 基本信息：标题、作者、年份、会议/期刊、链接、代码、数据、license。
2. 重要度：按对 Huawei RL 当前 research output 的直接价值排序。
3. 中文简介：一句话定位 + 解决的问题。
4. 内容核心：方法、数据、reward、训练、实验设置、主要结果。
5. 对本项目的启发：可迁移设计、可做实验、风险点。
6. 我的判断：值得复现 / 只作为背景 / 暂时观望。
7. 待办：下一次讨论或实验要做什么。

## 反向链接

- Huawei RL 文献库维护约定
- Huawei RL
- 14_Qwen_RLHF_Method_Ranking_Taskbook
- 01_Research_Output_Reward_Receipt
