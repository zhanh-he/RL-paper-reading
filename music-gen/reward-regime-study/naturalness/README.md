# 不自然人工感 reward 审计

最新：[外部证据与待办](EXTERNAL_EVIDENCE_HANDOFF.md) · [内部复验任务单](INTERNAL_REPLICATION_CHECKLIST.md) · [指标混杂审计](DESCRIPTOR_CONFOUND_AUDIT.md)。新增 [8次条件对比引导实测](CONTRAST_FOLLOWUP_REPORT.md)（含2次逐位一致sham，16个固定窗口）。仍无人工标签，artifact reward建议保持0；公共renderer的发现不能外推其他实现。

前轮：已完成[真实生成Prompt/CFG对照](PROMPT_CONTROL_REPORT.md)（32个分析输出、60个固定窗口），并提供[四组优先Suno试听](LISTENING_REVIEW_2026-11-05.md)。当前仍无可直接优化的电音reward。

最新主实验：[真实Suno V5音乐](SUNO_REAL_REPORT.md)，16首原始歌曲、48条未加工片段；先做真实听评定位，不直接优化未校准指标。

辅助：[47条心理声学实测](PSYCHOACOUSTIC_REPORT.md)已用RTX5090完成听觉子带调制、参考roughness与孤立共振比较；含正常合成器保护和失败对照。下文保留首轮25条AI scorer审计的复现记录。

[在线试听与评分](https://zhanh-he.github.io/RL-paper-reading/demos/#naturalness) · [实测报告](REPORT.md) · [19个候选与原始来源](RESEARCH.md) · [逐条分数 CSV](runs/2026-11-04/sample-scores.csv)

目前最推荐本域人工偏好 head，加 Audiobox PQ 质量约束。**尚不能作为已验证的 RL 自然度目标。** 25条10秒音频已跑通 Audiobox、PAM、SongEval和信号诊断；没有人工标签，没有本轮新增RL训练。自拟“自然／伪影”prompt对比已因饱和失败而排除。指定背景 Word 已完整阅读；其中伪影归因和检测断言已与一手资料分开核实，目标实现仍待代码/重建对照验证。

代码、数据和网页沿用现有 reward-regime-study 与 Music-RL Live。这里是一轮 scorer/perturbation audit，不是新的训练项目。

## 已存的证据

- [audit.json](runs/2026-11-04/audit.json)：公开清单、三模型分数、DSP、对照、方法与资源。
- [learned-scores.json](runs/2026-11-04/learned-scores.json)、[songeval-scores.json](runs/2026-11-04/songeval-scores.json)、[signal-scores.json](runs/2026-11-04/signal-scores.json)：未舍入结果和逐片段时间。
- [model-provenance.json](runs/2026-11-04/model-provenance.json)：固定代码/权重来源、哈希、许可证、失败修复。
- [human-agreement.json](runs/2026-11-04/human-agreement.json)：人工标签0；相关性、偏好准确率均为空。

全部25条在三个scorer、DSP、清单之间逐一匹配评分文件SHA256。原始来源、裁剪评分WAV、公开MP3分别记录哈希。网页只发布MP3 192kbps/48kHz，不作响度归一化；评分原件保存在repo之外。两条录制参考的上游本身为OGG，浮点解码不能恢复录音母带。CC-BY署名、原始网址与许可在每条清单中保留。

## 本机复现两个轻量 AI 评分器

从本目录执行。将 `PRIVATE_RUN` 改为repo之外的私有工作目录。以下模型/环境步骤可重新建立；本轮实际运行用的是已经核验的固定weights和锁定依赖。下载约1.1GB模型，不将权重提交到repo。

```bash
export PRIVATE_RUN=/absolute/path/to/private-audit
uv venv --python 3.11 "$PRIVATE_RUN/venv"
uv pip install --python "$PRIVATE_RUN/venv/bin/python" -r requirements.lock.txt
git clone https://github.com/soham97/PAM.git "$PRIVATE_RUN/PAM"
git -C "$PRIVATE_RUN/PAM" checkout ead6e7f9d907268e30c73b29526f50523a34b4aa
"$PRIVATE_RUN/venv/bin/python" download_models.py --directory "$PRIVATE_RUN/model-cache"
```

原件模式：保留本轮私有 `audit_manifest.json` 与对应 `scoring_audio/`，或从有权限的源文件用 `prepare_controls.py` 重建。它要求 `samples:[{id,label,path,model,checkpoint,input_id,pair_group,kind,rights,selection_reason}]`；`recorded_reference` 会加入第二组处理。外部参考可用 `fetch_public_references.py` 重取，CC-BY署名不可删除。原始下载源与全长哈希在公开清单可追溯。浮点WAV的容器元数据可能随重建变化，必须以当次manifest绑定文件，不能假设重建后文件哈希不变。

公开MP3模式：新checkout也能执行完整评分路径，但压缩波形不同，**不应逐数复现原件分数**。

```bash
"$PRIVATE_RUN/venv/bin/python" playback_manifest.py --directory "$PRIVATE_RUN/playback-scoring"
export AUDIT_MANIFEST="$PRIVATE_RUN/playback-scoring/playback-manifest.json"
# 原件模式把 AUDIT_MANIFEST 指向原件清单。
"$PRIVATE_RUN/venv/bin/python" signal_features.py --manifest "$AUDIT_MANIFEST" --output "$PRIVATE_RUN/signal.json"
"$PRIVATE_RUN/venv/bin/python" score_learned.py --manifest "$AUDIT_MANIFEST" --method audiobox --device cpu --threads 4 --aes-checkpoint "$PRIVATE_RUN/model-cache/audiobox-checkpoint.pt" --output "$PRIVATE_RUN/audiobox.jsonl"
"$PRIVATE_RUN/venv/bin/python" score_learned.py --manifest "$AUDIT_MANIFEST" --method pam --device cpu --threads 4 --pam-checkpoint "$PRIVATE_RUN/model-cache/msclap/CLAP_weights_2023.pth" --pam-source "$PRIVATE_RUN/PAM" --gpt2-dir "$PRIVATE_RUN/model-cache/gpt2" --embedding-dir "$PRIVATE_RUN/embeddings" --output "$PRIVATE_RUN/pam.jsonl"
"$PRIVATE_RUN/venv/bin/python" -m unittest discover -s . -p test_contract.py
```

输出JSONL是追加模式；重复跑请使用新的输出文件名，避免把旧/新运行混在一起。各文件旁写入初始化、依赖、权重和资源metadata。CPU路径在本机实测；MPS/CUDA未在本轮测量。

[SongEval复现说明](SONGEVAL_REPRODUCTION.md)使用单独的已有Linux CPU环境，避免破坏PAM所需旧Transformers版本。其 Naturalness 主要衡量人声呼吸/分句，Clarity主要衡量歌曲结构，不能把名字当全局人工音色检测。

## 人工标签与可训练的小 head

网页默认隐藏来源和自动分，不填任何假MOS。分别记录不自然人工感、音质、音乐性和三维A/B偏好，允许相当/无法判断。数据仅保存在浏览器本机，导出JSON后再分析。固定编号和手动A/B辅助不等于正式随机盲测。

```bash
python evaluate_labels.py --audit runs/2026-11-04/audit.json --labels listener1.json listener2.json --output agreement.json
```

这个命令只用真实人工列；没有标签时输出 `no_human_labels` 和空指标。反例处理不充当自然度gold。比较准确率限制为同条件/同来源对，CI按来源组bootstrap；不足3组不填CI。少样本结果只作探索。

`preference_head.py`提供冻结embedding＋L2线性Bradley–Terry训练和推理函数。输入NPZ：`ids`为唯一字符串数组，`embeddings`为对应二维矩阵；本轮PAM每条两窗1024维embedding已在私有目录保存，先按片段求均值。训练标签是明确转换后的 `pairs:[{a,b,choice:"a"|"b"|"tie",split:"train"|"test"}]`。单独提供源注册表 `samples:{id:{song_group,prompt_group,split}}`，确保歌曲/prompt/ID跨集合不重叠。网页导出中的 `A/B` 需转成这里的小写，`uncertain`不训练；未提供自动标签转换以避免误把空值当偏好。

```bash
python preference_head.py --embeddings embeddings.npz --pairs verified-pairs.json --split-registry source-split.json --encoder-id microsoft-msclap-2023 --encoder-revision c47d441165daa21986ead0850660917636a81775 --output head.json
```

本轮没有训练head。代码的至少10对只是调用约束，远非泛化充分性。建议先约40独立prompt×3对×2听者，按组24/8/8分train/calibration/test，并留出生成模型。正常合成器、明亮原声、电子乐、休止和人声都要覆盖。

## 与现有 RL 接口

[`reward_adapter.py`](reward_adapter.py)是可执行的组合示例，尚未接入现有trainer。它显式将existing、head logit、PQ按冻结校准集p05/p95重映射到[-1,1]，会改变旧reward尺度；不能直接替换后声称训练已保持一致。

```python
from preference_head import predict
from reward_adapter import CalibratedReward

# ranges必须来自独立、真实标签审核过的校准集，不能用本轮25条凑假标定。
calibrated = CalibratedReward(**ranges, weight=0.05)
logit = predict(embedding, trained_head, encoder_id=encoder_id, encoder_revision=revision)
row = calibrated.compose(existing=old_reward, naturalness_logit=logit, pq=pq,
    valid=audio_valid, activity_ok=task_coverage_ok, duration_ok=duration_ok,
    clipping_fraction=full_scale_fraction, shared_guard_already_applied=True)
```

shadow阶段只记录，不调用该组合改变训练。校准/听测通过后，再扫小权重并做同起点、同预算的真正对照训练。现有richness/instrument/arrangement已减shared guard；Mean7路径不同，必须检查实际arm，指定唯一guard归属，防止削波惩罚重复。`saturated`与`unclipped_reward`用于发现裁剪饱和。整条静音/截短/失败拒绝，正常局部休止不自动判坏。音乐性、歌词对齐、节奏、多样性和配器继续独立监控。

## 网页维护

`build_audit.py`只从同一私有运行目录合并实际结果，拒绝跨scorer hash错配；公开字段使用allowlist。`publish_audio.py`执行MP3编码和独立hash登记；不上传原件。`platform/scripts/sync-naturalness.mjs`将公开snapshot同步到现有网页，检查25条分数、0人工标签与MP3绑定。

```bash
npm --prefix platform run build
npm --prefix platform run check
```

这些命令从repo根执行。发布沿用现有GitHub Pages workflow。背景文档已读；人工确认和目标系统上的泛化验证仍待完成，报告和网页明确保留此状态。
