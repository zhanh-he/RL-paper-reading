#实验机交接 Prompt

```text
你在团队实验机上执行 YuE2 歌手身份与任务 reward 实验。先读取 https://github.com/zhanh-he/RL-paper-reading/tree/main/notes/2026-11-17-yue2-taskwise-reward 中的 README.md、REWARDS.md、PROTOCOL.md、prompt_seeds.jsonl 和 annotations.csv。若 GitHub 访问受限，同一份文件可由我作为附件发给你。策划资料在 RL-paper-reading，实际运行使用实验机上团队的真实 YuE2 代码与模型。

先自己定位可用仓库/commit、checkpoint、推理入口、现有 reward 及训练入口、GPU、输出根目录。记录实际 sampler、KL、可训练模块和 mix/stem 接口。跑少量 smoke，测时并固定设置，优先复用已有自然漂移样本。按协议生成独唱开发/确认集并保存 manifest，开发与确认按曲目隔离。附上纯音乐和 A/B/A 轮唱的小探针，遇到模型接口不支持时给出实际证据。把音频、标注、运行日志和 checkpoint 保存到实验机的团队实验区。

实现或接入四种身份打分：不重叠分窗平均 cosine、长距离 cosine Q10、medoid 锚点低分位、低相似异常持续占比。同步记录现用 reward、可听人声、歌词覆盖、有效音乐、评分状态及异常时间。对同一条件保留所有候选，记录渲染重复性。完成这里后，先给我一个匿名盲听包和 annotations.csv，告诉我在哪里从 mix 挑自然换人、正常换声区、漏唱和不确定例。请等我回填标签，届时再用开发集选配置、用确认集做重排验证。第一份报告给出路径、真实运行量、已知失败和我该听哪些文件。

收到标签后，按协议用相同输入做离线重排。确认达到写明的门槛后才跑独唱 A/B GRPO，训练保持相同起点、采样及预算，B 只切换身份项。对冻结的留出集生成匿名 A/B 包，等我完成最终盲听并汇报身份、歌词、伴奏、表现力和不确定性。依据协议结果决定是否进入混合任务；同时合唱先做专项评判器验证。全程保存样本 ID、配置、证据和可复现命令，遇到外部权限或缺失资源时报告具体路径和可复现错误。
```
