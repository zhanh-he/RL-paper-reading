# MusicGen Live · 方法论探索的维护与发布

在线入口为 `/musicgen/`。原 `/demos/` 改名 **Music-RL Live**，继续保存RL、reward、GRPO与历史回放，旧URL和锚点不变。AMT仍在 `/amt/demos/`。三个页面互链，不能把旧RL结果算进本页的新方法实验。

## 数据源与页面

- `platform/data/musicgen/experiments.json` 是本页唯一手动维护的计划与公开摘要源。
- `platform/scripts/sync-musicgen-experiments.mjs` 严格校验并生成 `platform/site/musicgen/data/experiments.json` 和 `snapshot.json`；后者记录source与public JSON各自SHA256。
- `platform/site/musicgen/{index.html,styles.css,app.js}` 实现页面。浏览器只读本站静态数据，不连接训练服务器，也不接收原始内网结果文件。
- 文献仍以RL/AMT CSV为准，本页参考链接不复制评分或私有阅读笔记。

本页追踪条件生成、显式符号计划、AR/NAR、数据契约与控制评测。CoT指音乐中间计划。局部音频编辑、冻结AR真实草稿混合训练是后续方案；已有工程实现与未来能力分开标记。

## 怎样记录计划和结果

`hypotheses`、`capabilities`、`ablations`、`protocol`、`roadmap`描述假设和待验证方案。当前2×2消融仍待预登记，不能因为已有pilot模板就改成completed。

`evidence`只放经审核的工程摘要。2026-10-26基础双任务各150项测试与2026-10-27另一探索范围327项测试必须保留各自日期、scope和局限，不能相加或用最新计数覆盖历史。CPU训练次数不代表GPU训练或音乐质量。

`training_results`目前为空。未来每项仅允许：

```json
{
  "id": "anonymous-run-id",
  "experiment_id": "registered-experiment-id",
  "date": "2026-10-29",
  "task": "vocal2accomp",
  "evidence_kind": "training_measurement",
  "status": "completed",
  "config_id": "public-config-identity",
  "split_id": "public-song-split-identity",
  "seed": 42,
  "metrics": [],
  "summary": "仅说明真实完成的范围；此处是格式示例，不是实验结果。"
}
```

`task`仅为vocal2accomp/lyrics2song；`evidence_kind`为training_measurement/quality_measurement；`status`为completed/failed。匿名身份须能在原机器追溯，但不能包含内部路径、原曲或机器身份。需要更完整的预算与配置时，另冻结经审核的公开协议，再扩展schema，不能往结果任意塞字段。

每个metric只允许 `id/value/n/unit/direction/scope`。value为有限数或null，n为正整数或null；有数值时必须有正样本数。scope为training_diagnostic/quality/resource，训练诊断不能标成音质证据。completed的quality_measurement至少有一个实测quality指标；failed结果不附评分。主观评价要说明听测设计及样本数，不能用loss或自动评价器总分直接宣称人类更偏好。

回传顺序：实验机现有工具导出标准TXT/JSON → 在受控电脑校验schema和hash → 人工审核公开范围 → **仅转写允许字段到此公开schema** → 构建检查 → 发布。内部完整回执仍留原处。source/public SHA256只能检查内容一致性，不证明实验真实性或审核者身份。

禁止发布私有源码、完整日志、内部路径、账号凭证、host/job身份、歌词/提示词/乐谱/音频样本、checkpoint或未经许可的素材。程序校验是辅助，不能代替人工审核。当前本页无音频；以后若获准加试听，遵守仓库MP3-only策略。

## 检查与部署

```bash
npm --prefix platform run build
npm --prefix platform run check
npm --prefix platform run test:musicgen
npm --prefix platform run test:catalog
```

GitHub Pages沿用现有workflow。提交前复核只改本任务文件，保持CSV、请求inbox、旧实验结果和音频不变；推送后确认workflow与三个入口。更新静态快照不授权启动训练、提交集群任务或建立自动监控。
