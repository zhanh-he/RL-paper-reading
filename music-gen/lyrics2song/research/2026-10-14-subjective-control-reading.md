# 主观与弱可验证控制：跨视觉、语音与音乐的阅读地图

核验日期：2026-10-14。这里只整理公开文献事实和方法边界，不发布私人报告全文、内部模型结果或本地知识库。不是SOTA榜，也不是新实验记录。

## 结论

- “能算分”“测对构念”“优化分数后人类更喜欢”是三个不同命题。离线排序准确不足以保证policy更新后的效用。
- DPO适合可审计的成对偏好；刷新候选可以用semi-online DPO，不必接GRPO。DPO本身不免疫噪声或both-bad pairs。
- online RL需要当期输出可可信评价；RSFT/OPD提供更密集的保留信号，但不是自动的局部credit assignment。
- differentiable reward和DiffusionNFT改变更新接口，不自动修正错误reward。连续DiT/flow与AR token policy分开比较。
- hard/soft可分解，但scalar sum允许补偿；期望约束不等逐首veto，检测器confidence/评委共识不等正确率校准。

延伸：[BRPO / NFT 决断](2026-10-14-brpo-nft-reward-validity.md) · [LeVo2 / YuE2 / Full-Song 核验](2026-10-14-post-training-reading.md) · [交互文献目录](https://zhanh-he.github.io/RL-paper-reading/)

## 新增的直接参考

| 文献 | 值得借鉴 | 不能直接推出 |
|---|---|---|
| [GSRM](https://arxiv.org/abs/2602.13891) | feature-grounded speech RM；§5/D.2有speech actor GRPO，E.2另有judge GRPO | speech听感收益不是full-song beat utility；20/eight judge预算描述冲突 |
| [VIBE](https://arxiv.org/abs/2608.30125) | hard tempo/key + soft CMI/omni、五阶段NFT；官方项目标Findings EMNLP2026 accepted | 10s器乐不等全曲；半/倍拍容忍不等vocal-accomp同步；加和不保证非劣 |
| [RRPO](https://arxiv.org/abs/2512.04552) | emotional TTS可微reward被acoustic artifacts利用；robust RM与hybrid regularization | 不是GRPO专属修复，也不是DPO天然更抗噪 |
| [FPO](https://doi.org/10.1109/TASLPRO.2025.3648888) | 局部segmental errors与selective loss；正式卷34、557–566，2026 | 总和section reward广播全tokens不等局部更新 |
| [UNO](https://arxiv.org/abs/2406.00654) | 保留主观不确定性，不将审美分歧全当错误 | taste方差不等beat方向错误概率 |
| [MPO](https://arxiv.org/abs/2509.00685) | 多维preferred/dispreferred集合与保留项 | 随机跨集合pair不保证逐轴Pareto支配 |
| [DiffRO](https://arxiv.org/abs/2507.05911) | token reward与固定renderer输出评分可脱钩 | proxy可微不代表真实音频效用可微 |

## 必须带上的数学限定

1. 标准化advantage只对正仿射reward变换近似不变；一般单调非线性变换不具同样保证。G=2非tie接近±1的特殊情形另列。
2. 低std不一定意味着弱更新：精确同分为0，但tiny nonzero gap可被std放大。数值稳定ε与PPO clip ε是不同参数。
3. 高满足度V的约束E[V]≥τ，可用max-policy/min-nonnegative-dual的Lagrangian；仍是平均约束，不是每首歌必须通过。
4. 可微pathwise梯度需要固定reward、独立基础噪声的可微重参数化与允许交换期望/梯度；黑盒madmom不能直接穿透。
5. 偏好熵衡量共识程度，不是判向正确率。低样本量与合法taste异质性不能忽略。
6. RSFT/OPD密集token loss、歌曲时间局部反馈与SPO denoising-step偏好是不同粒度。

## 结果与版本校订

**Towards Hallucination-Free Music v1**：[原文](https://arxiv.org/html/2508.05011v1)。Table1验证平均reward为Origin .771、DPO .843、RS+DPO .845、PPO .820、GRPO .834，因此绝对分差为.072/.074/.049/.063。摘要7.4/4.9/4.7不完全一致；不写成独立人评PER相对下降，也不用作通用算法胜负。§3.6实际GRPO去掉std，与通用标准化式不同。

**AlignProp**：原arXiv已撤回并被[Video Diffusion Alignment via Reward Gradients](https://arxiv.org/abs/2407.08737)吸收；原项目可读，但不是已核实独立ICLR发表。

**DLPO**：主引用为2024预印本2405.14632；2025同作者另题DLPO稿已在[Interspeech](https://www.isca-archive.org/interspeech_2025/chen25b_interspeech.html)发表，并另有2508.03123身份。数字与作者列表按版本归属，不默认为同一稿。

## 原始文献与官方资源

评分不代填。新文献保留paper / code / demo / web；未知入口留空。第一作者机构及已核实量化规模在交互目录中；没有跨论文统一速度排名。

| 年份 / Venue | 完整文献 | 官方资源 | Catalog |
|---|---|---|---|
| 2024<br>ICLR | [Training Diffusion Models with Reinforcement Learning](https://arxiv.org/abs/2305.13301) | [github](https://github.com/kvablack/ddpo-pytorch) · [demo](https://rl-diffusion.github.io/) · [web](https://proceedings.iclr.cc/paper_files/paper/2024/hash/14f75513f0f1ca01de1e826b52e6b840-Abstract-Conference.html) | [条目](https://zhanh-he.github.io/RL-paper-reading/?paper=2024-ICLR-DDPO-2305.13301) |
| 2023<br>NeurIPS | [ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation](https://arxiv.org/abs/2304.05977) | [github](https://github.com/zai-org/ImageReward) · [web](https://proceedings.neurips.cc/paper_files/paper/2023/hash/33646ef0ed554145eab65f6250fab0c9-Abstract-Conference.html) | [条目](https://zhanh-he.github.io/RL-paper-reading/?paper=2023-NeurIPS-ImageReward-2304.05977) |
| 2024<br>CVPR | [Diffusion Model Alignment Using Direct Preference Optimization](https://arxiv.org/abs/2311.12908) | [github](https://github.com/SalesforceAIResearch/DiffusionDPO) · [web](https://openaccess.thecvf.com/content/CVPR2024/html/Wallace_Diffusion_Model_Alignment_Using_Direct_Preference_Optimization_CVPR_2024_paper.html) | [条目](https://zhanh-he.github.io/RL-paper-reading/?paper=2024-CVPR-Diffusion-DPO-2311.12908) |
| 2023<br>arXiv (withdrawn) | [Aligning Text-to-Image Diffusion Models with Reward Backpropagation](https://arxiv.org/abs/2310.03739) | [github](https://github.com/mihirp1998/AlignProp) · [demo](https://align-prop.github.io/) · [web](https://arxiv.org/abs/2407.08737) | [条目](https://zhanh-he.github.io/RL-paper-reading/?paper=2023-arXiv-AlignProp-2310.03739) |
| 2025<br>NeurIPS | [Flow-GRPO: Training Flow Matching Models via Online RL](https://arxiv.org/abs/2505.05470) | [github](https://github.com/yifan123/flow_grpo) · [demo](https://gongyeliu.github.io/Flow-GRPO/) | [条目](https://zhanh-he.github.io/RL-paper-reading/?paper=2025-NeurIPS-Flow-GRPO-Flow-Matching-Online-RL) |
| 2025<br>arXiv | [Emotion-Director: Bridging Affective Shortcut in Emotion-Oriented Image Generation](https://arxiv.org/abs/2512.19479) | 未核实独立代码/demo | [条目](https://zhanh-he.github.io/RL-paper-reading/?paper=2025-arXiv-Emotion-Director-2512.19479) |
| 2024<br>arXiv | [Enhancing Zero-shot Text-to-Speech Synthesis with Human Feedback](https://arxiv.org/abs/2406.00654) | 未核实独立代码/demo | [条目](https://zhanh-he.github.io/RL-paper-reading/?paper=2024-arXiv-UNO-2406.00654) |
| 2025<br>ICASSP | [Emo-DPO: Controllable Emotional Speech Synthesis through Direct Preference Optimization](https://arxiv.org/abs/2409.10157) | [web](https://doi.org/10.1109/ICASSP49660.2025.10888737) | [条目](https://zhanh-he.github.io/RL-paper-reading/?paper=2025-ICASSP-Emo-DPO-2409.10157) |
| 2024<br>arXiv | [DLPO: Diffusion Model Loss-Guided Reinforcement Learning for Fine-Tuning Text-to-Speech Diffusion Models](https://arxiv.org/abs/2405.14632) | [web](https://www.isca-archive.org/interspeech_2025/chen25b_interspeech.html) | [条目](https://zhanh-he.github.io/RL-paper-reading/?paper=2024-arXiv-DLPO-2405.14632) |
| 2025<br>NCMMSC | [MPO: Multidimensional Preference Optimization for Language Model-based Text-to-Speech](https://arxiv.org/abs/2509.00685) | [demo](https://anonymous-person01.github.io/MPO-demo/) | [条目](https://zhanh-he.github.io/RL-paper-reading/?paper=2025-NCMMSC-MPO-2509.00685) |
| 2026<br>TASLP | [FPO: Fine-Grained Preference Optimization Improves Zero-Shot Text-to-Speech](https://arxiv.org/abs/2502.02950) | [demo](https://yaoxunji.github.io/fpo/) · [web](https://doi.org/10.1109/TASLPRO.2025.3648888) | [条目](https://zhanh-he.github.io/RL-paper-reading/?paper=2026-TASLP-FPO-2502.02950) |
| 2026<br>arXiv | [GSRM: Generative Speech Reward Model for Speech RLHF](https://arxiv.org/abs/2602.13891) | 未核实独立代码/demo | [条目](https://zhanh-he.github.io/RL-paper-reading/?paper=2026-arXiv-GSRM-2602.13891) |
| 2025<br>NeurIPS AIforMusic Workshop | [Generative Multi-modal Feedback for Singing Voice Synthesis Evaluation](https://arxiv.org/abs/2512.02523) | [github](https://github.com/opendilab/VocalCritic) | [条目](https://zhanh-he.github.io/RL-paper-reading/?paper=2025-NeurIPS-Workshop-VocalCritic-2512.02523) |
| 2024<br>ICML | [MusicRL: Aligning Music Generation to Human Preferences](https://proceedings.mlr.press/v235/cideron24a.html) | [demo](https://google-research.github.io/seanet/musiclm/rlhf/) | [条目](https://zhanh-he.github.io/RL-paper-reading/?paper=2024-ICML-MusicRL-Aligning-Music-Generation-to-Human-Preferences) |
| 2025<br>IJCAI | [NotaGen: Advancing Musicality in Symbolic Music Generation with Large Language Model Training Paradigms](https://arxiv.org/abs/2502.18008) | [github](https://github.com/ElectricAlexis/NotaGen) · [demo](https://electricalexis.github.io/notagen-demo/) | [条目](https://zhanh-he.github.io/RL-paper-reading/?paper=2025-IJCAI-NotaGen-Advancing-Musicality-in-Symbolic-Music-Generation) |
| 2025<br>arXiv | [Towards Hallucination-Free Music: A Reinforcement Learning Preference Optimization Framework for Reliable Song Generation](https://arxiv.org/abs/2508.05011) | 未核实独立代码/demo | [条目](https://zhanh-he.github.io/RL-paper-reading/?paper=2025-arXiv-Towards-Hallucination-Free-Music) |
| 2026<br>ICASSP | [MR-FlowDPO: Multi-Reward Direct Preference Optimization for Flow-Matching Text-to-Music Generation](https://arxiv.org/abs/2512.10264) | [github](https://github.com/lonzi/mrflow_dpo) · [demo](https://lonzi.github.io/mr_flowdpo_demopage/) | [条目](https://zhanh-he.github.io/RL-paper-reading/?paper=2026-ICASSP-MR-FlowDPO) |
| 2026<br>arXiv | [LeVo 2: Stable and Melodious Song Generation via Hierarchical Representation Modeling and Progressive Post-Training](https://arxiv.org/abs/2606.30642) | [github](https://github.com/levo-demo/LeVo) · [demo](https://levo-demo.github.io/levo_v2_demo/) | [条目](https://zhanh-he.github.io/RL-paper-reading/?paper=2026-arXiv-LeVo-2-2606.30642) |
| 2026<br>arXiv | [Pushing the Frontier of Full-Song Generation: Hierarchical Autoregressive Planning Meets Flow-Matching Rendering](https://arxiv.org/abs/2607.20253) | [demo](https://hifi-song-generation.github.io/) | [条目](https://zhanh-he.github.io/RL-paper-reading/?paper=2026-arXiv-Alibaba-Token-Foundry-2607.20253) |
| 2026<br>arXiv | [YuE2: Unifying Symbolic and Audio Music Generation at Frontier Quality](https://arxiv.org/abs/2609.33757) | [github](https://github.com/multimodal-art-projection/YuE) · [demo](https://map-yue2.github.io/) | [条目](https://zhanh-he.github.io/RL-paper-reading/?paper=2026-arXiv-YuE2-2609.33757) |
| 2026<br>Findings of EMNLP (accepted) | [VIBE: Video Instruction-aligned Background music gEneration](https://arxiv.org/abs/2608.30125) | [github](https://github.com/aryanvibhosale/vibe) · [demo](https://vibe-text-video-to-music-generation.github.io/vibe/) | [条目](https://zhanh-he.github.io/RL-paper-reading/?paper=2026-EMNLP-Findings-VIBE-2608.30125) |
| 2023<br>NeurIPS | [Direct Preference Optimization: Your Language Model Is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) | 未核实独立代码/demo | [条目](https://zhanh-he.github.io/RL-paper-reading/?paper=2023-NeurIPS-Direct-Preference-Optimization) |
| 2024<br>arXiv | [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) | [github](https://github.com/deepseek-ai/DeepSeek-Math) · [web](https://arxiv.org/html/2402.03300v3) | [条目](https://zhanh-he.github.io/RL-paper-reading/?paper=2024-arXiv-DeepSeekMath-GRPO-2402.03300) |
| 2025<br>arXiv | [Differentiable Reward Optimization for LLM based TTS system](https://arxiv.org/abs/2507.05911) | 未核实独立代码/demo | [条目](https://zhanh-he.github.io/RL-paper-reading/?paper=2025-arXiv-DiffRO-2507.05911) |
| 2025<br>CVPR | [Aesthetic Post-Training Diffusion Models from Generic Preferences with Step-by-step Preference Optimization](https://arxiv.org/abs/2406.04314) | [github](https://github.com/RockeyCoss/SPO) · [demo](https://rockeycoss.github.io/spo.github.io/) · [web](https://openaccess.thecvf.com/content/CVPR2025/html/Liang_Aesthetic_Post-Training_Diffusion_Models_from_Generic_Preferences_with_Step-by-step_Preference_CVPR_2025_paper.html) | [条目](https://zhanh-he.github.io/RL-paper-reading/?paper=2025-CVPR-SPO-2406.04314) |
| 2026<br>ICLR (Oral) | [DiffusionNFT: Online Diffusion Reinforcement with Forward Process](https://arxiv.org/abs/2509.16117) | [github](https://github.com/NVlabs/DiffusionNFT) · [demo](https://research.nvidia.com/labs/cosmos-lab/diffusionnft/) · [web](https://arxiv.org/html/2509.16117v2) | [条目](https://zhanh-he.github.io/RL-paper-reading/?paper=2026-ICLR-DiffusionNFT-2509.16117) |
| 2026<br>ICASSP (accepted) | [RRPO: Robust Reward Policy Optimization for LLM-based Emotional TTS](https://arxiv.org/abs/2512.04552) | [demo](https://lrwinr.github.io/RRPO-CosyVoice/) | [条目](https://zhanh-he.github.io/RL-paper-reading/?paper=2026-ICASSP-RRPO-2512.04552) |
| 2026<br>ICML | [CMI-RewardBench: Evaluating Music Reward Models with Compositional Multimodal Instruction](https://arxiv.org/abs/2603.00610) | [github](https://github.com/Haiwen-Xia/CMI-RewardBench) · [web](https://huggingface.co/datasets/HaiwenXia/cmi-pref) | [条目](https://zhanh-he.github.io/RL-paper-reading/?paper=2026-arXiv-CMI-RewardBench-and-CMI-RM) |
| 2025<br>arXiv | [SongEval: A Benchmark Dataset for Song Aesthetics Evaluation](https://arxiv.org/abs/2505.10793) | [github](https://github.com/ASLP-lab/SongEval) · [demo](https://aslp-lab.github.io/SongEval/) · [web](https://huggingface.co/datasets/ASLP-lab/SongEval) | [条目](https://zhanh-he.github.io/RL-paper-reading/?paper=2025-arXiv-SongEval-Song-Aesthetics-Evaluation) |
| 资源<br>官方评测资源 | [SongBench](https://github.com/Tencent/SongBench) | [github](https://github.com/Tencent/SongBench) | 官方资源 |

## 下一步讨论的优先级

高优先：构念资格、current-policy方向可靠性、chosen绝对质量、非目标维度保留、独立人评协议。随后才讨论semi-online DPO、RSFT/OPD或合适接口的BRPO/NFT对照。情绪反事实、局部reward、多模态guidance是后续设计线索，不是本次启动任务。

本轮不改变既有split、sealed test、训练配置、作业或发布音频策略。

