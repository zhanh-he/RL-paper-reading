# 外部条件对比引导：实测补充

2026-11-05。**工程控制已验证；听感效果待评价，不能用于证明artifact reward或GRPO改善。**

使用 DiffRhythm2 官方公开权重、同一套自写歌词、两种配器、seed37，真实生成8次60秒（含2次sham）。分支均保留相同歌词与生成历史，只改变风格条件。公式 `v_positive + β(v_positive − v_avoid)`，β=0/.5/1；这是两条件外推，区别于原生清空歌词和风格的CFG。提示写了自然/机器人不等于生成真的好/坏。

已核查：初始噪声序列一致；歌词前缀实际tensor一致；2/2 sham的latent和解码PCM逐位一致；共享源码未修改。固定10–20s与40–50s评分，原音量，无人工添加伪影。声压级统一70dB是DW分析假设；播放器MP3未做音量归一。

| 配器 | 条件 | 全带M | <1.5k中心组M | 3–8k中心组M | DW70 | 保护共振 | PQ监测 |
|---|---|---:|---:|---:|---:|---:|---:|
| acoustic | conditional | 0.2854 | 0.2796 | 0.3498 | 0.0412 | 0.0063 | 8.2110 |
| acoustic | contrast05 | 0.2540 | 0.2515 | 0.3217 | 0.0378 | 0.0025 | 8.4165 |
| acoustic | contrast10 | 0.2211 | 0.2188 | 0.2739 | 0.0367 | 0.0013 | 8.4095 |
| acoustic | sham10 | 0.2854 | 0.2796 | 0.3498 | 0.0412 | 0.0063 | 8.2110 |
| synth | conditional | 0.3428 | 0.3234 | 0.3987 | 0.0925 | 0.0013 | 7.5629 |
| synth | contrast05 | 0.3171 | 0.2971 | 0.3936 | 0.1160 | 0.0018 | 7.8862 |
| synth | contrast10 | 0.3097 | 0.2927 | 0.3926 | 0.1147 | 0.0017 | 7.7783 |
| synth | sham10 | 0.3428 | 0.3234 | 0.3987 | 0.0925 | 0.0013 | 7.5629 |

表中为两个固定窗口均值，不是整曲评估；没有已验证的“指标越低越自然”方向。分带M按听觉子带中心分组，频率响应重叠，并非互斥硬频带。

本轮修正先前acoustic具体提示中出现synth词汇的混杂：acoustic两分支均不提合成器，synth两分支均明确保留亮的模拟琶音、pad、电子bass与鼓机。不过措辞仍同时改变多项音色属性，只能解释为组合条件干预。只有一个seed；每个非sham干预未多次重复，也未验证更多歌词与内容，不能估计泛化收益。

[试听与评分](https://zhanh-he.github.io/RL-paper-reading/demos/?nat-corpus=contrast&nat-pair=contrast-synth-contrast05#naturalness) · [代码](generate_diff_contrast.py) · [登记输入](runs/2026-11-05-contrast/batch.json) · [逐样本与有效参数](runs/2026-11-05-contrast/audit.json) · [实际调用证据](runs/2026-11-05-contrast/receipt.json)

来源：[官方DiffRhythm2](https://github.com/xiaomi-research/diffrhythm2)；公开权重revision与SHA、实际耗时和显存见receipt。部署环境与缓存位置由运行者提供，源码/权重不会拷入本repo。

复现（先配置官方依赖与本地授权资产）：
```sh
python generate_diff_contrast.py --source "$DIFF_SOURCE" --assets "$DIFF_ASSETS" --runtime-root "$DIFF_RUNTIME" --batch runs/2026-11-05-contrast/batch.json --output "$RUN_OUTPUT" --seconds 60 --steps 32 --threads 4
python prepare_generation_excerpts.py --runs "$RUN_OUTPUT/run.json" --output-dir "$CLIPS" --manifest "$EXCERPTS"
python auditory_fulltrace.py --manifest "$EXCERPTS" --output "$AUDITORY" --device cuda --skip-roughness
OPENBLAS_NUM_THREADS=1 python score_roughness70.py --manifest "$EXCERPTS" --output "$ROUGHNESS" --workers 4
python score_learned.py --manifest "$EXCERPTS" --output "$QUALITY" --method audiobox --aes-checkpoint "$AUDIOBOX_WEIGHTS" --device cpu --threads 4
```

还需将本轮登记输入保存为 `contrast_batch.json`，并将评分原WAV、试听MP3及完整生成MP3置于该私有工作目录的 `clips/`（文件名保持case/window ID）。将同一轮run/excerpts/auditory/roughness JSON与audiobox JSONL命名为 `run.json`、`excerpts.json`、`auditory.json`、`roughness.json`、`audiobox.jsonl` 置于单个私有工作目录，再运行 `python build_contrast_followup.py --workspace "$PRIVATE_WORK" --output "$PUBLIC_JSON"`。原始WAV保留在repo之外；公开MP3另有编码指纹。

实测生成加载38.8s；8次生成合计106.1s；峰值本进程CUDA已分配5.12GiB（非整卡独占资源）；分析耗时见audit。两次sham不是额外独立音乐样本。
