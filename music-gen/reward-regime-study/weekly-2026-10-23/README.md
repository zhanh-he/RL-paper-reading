# 本周汇报：RL 与音乐 Reward

2026-10-23 AWST。推荐主线：**后训练的响应强烈依赖模型和reward；优化一个可测代理，不等于提高人类音乐质量。**

[在线图册与新试听](https://zhanh-he.github.io/RL-paper-reading/demos/weekly-2026-10-23.html) · [当前局部放大图下载](https://zhanh-he.github.io/RL-paper-reading/demos/weekly-2026-10-23/progression-focused.zip) · [逐条件结果](summary.json) · [本轮启动与回收说明](FOLLOWUP_2026-10-23.md)

## 新版：按训练步数画三模型折线

[三模型折线图](https://zhanh-he.github.io/RL-paper-reading/demos/weekly-2026-10-23.html#progression) · [仅局部放大图下载](https://zhanh-he.github.io/RL-paper-reading/demos/weekly-2026-10-23/progression-focused.zip) · [数据 CSV](progression.csv) · [来源及图文件哈希](progression.json)

DiffRhythm2、YuE2、Muse各一张，横轴为真实数值step 0/50/100/270，纵轴只看同批音频的SongBench七维均值；step0冻结baseline画水平虚线。按最新要求仅显示C局部放大版，旧A/B与端点图从页面移除，保留历史下载URL。默认只画SP、Mean7及真正SongEval训练；Instrument/Arrangement保留在完整表，不混作Mean7训练臂。

现有结果仍不足以绘制“三reward × train/valid/test”的九根实测曲线：完整固定split仅valid30；在线train batch不冒充固定train评估，full test未完成。小LR训练已提交/进入依赖链，但没有完成分数。Muse旧SP仅0/270完整，50/100为n/a；新版重跑补齐checkpoint。`progression.csv`对缺失train/test保留空值与unavailable状态，不复制valid，不补零。

本轮新增真正YuE2-SongEval3e-4的120/120交叉评分，四个Mean7为5.271/5.364/5.443/5.561。它保留自己的旧采样baseline，与SP轮不是同一份音频，图上单独画紫色基准线并只报告各自变化，不能当作严格reward单变量消融。见`songeval-cross.json`的每条件七头分数、音频SHA和评分器身份；并非改名的SongBench低LR臂。

`build_progression.py`从`summary.json`和已核验的`songeval-cross.json`逐条件七个原始头独立重算所有绘制均值，导出三模型局部放大PNG/PDF/SVG、CSV/JSON与ZIP。既有原始审计及旧端点图不改写。`collect_followup.py`是手动执行的一次性回收器，没有周期任务；`next-experiments.json`仅为最后核查快照。

## 建议怎么讲

1. **问题与协议，约一分钟。** 不做SOTA排行榜；研究不同reward在相同模型内部造成什么变化。冻结CMI-Pref270 train /30 valid，所有主结果用30/30固定验证和270步终点，不拿训练batch均值当验证，不拿Mock填数。270条件能观察到响应，但尚不能推出“小数据足以稳定提高完整歌曲质量”。
2. **SongEval确实能改变验证分数，但证据有边界。** 放图01-A或01-C，随后可截原demo的Reward/KL。YuE2-SongEval3e-4的验证均值3.464→3.661，配对Δ+0.197 [0.043,0.354]；2e-5和1e-4约+0.060/+0.072，区间包含0。强调一个训练seed、LR探索未校正、短600-token样本；不是人类MOS提升结论。
3. **SP代理与音乐评分会冲突。** 放图02-A与03-A。YuE2 raw SP0.217→0.531，Δ+0.314 [0.248,0.379]，而SongBench Mean7为5.391→5.165，Δ−0.226 [−0.450,−0.014]。这是“代理迁移不保证”的直接例子，不能仅凭这些分数宣称听感变差或已经证实reward hacking。先展示同一固定输入的音频，再讨论是否出现可听退化。
4. **SongBench不是换上去就一定有效。** YuE2的Instrument/Arrangement分别对应Mean7变化+0.612 [0.474,0.751]、+0.356 [0.193,0.526]。但Mean7包含被优化头，这不是独立裁判。新增DiffRhythm2纯Mean7单reward完成270步，5.116→5.055，Δ−0.061 [−0.179,0.052]，没有确立改善。用图04看各头，不从训练曲线就下结论。
5. **收尾：讲清新增与缺口。** 新增原始Muse-SP完成270步，raw SP0.060→0.233，Δ+0.173 [0.109,0.235]；Mean7为3.971→4.064，Δ+0.093 [−0.172,0.368]，迁移增益未确立。Muse两个learned-head任务启动失败；新YuE2-TRL Mean7验收失败，不能说这三组已训练成功。下一步先修这些接口并补独立评价/盲听，再做匹配时长、LoRA与β的多seed比较。

## 核心表

所有Mean7均为同批完整30条验证音频的七头原始均值，**不是test、人类MOS或独立听感证据**。

| 模型 | 优化目标 | Mean7 baseline→270 | 配对Δ [95% CI] |
| --- | --- | --- | --- |
| YuE2 | SP减guard |5.391→5.165|−0.226 [−0.450,−0.014]|
| YuE2 | Instrument/10减guard |5.391→6.003|+0.612 [0.474,0.751]|
| YuE2 | Arrangement/10减guard |5.391→5.747|+0.356 [0.193,0.526]|
| DiffRhythm2 | SP减guard |5.116→5.077|−0.039 [−0.114,0.032]|
| DiffRhythm2 | Instrument/10减guard |5.116→5.161|+0.045 [−0.074,0.183]|
| DiffRhythm2 | Arrangement/10减guard |5.116→5.228|+0.112 [0.025,0.204]|
| Muse | SP减guard |3.971→4.064|+0.093 [−0.172,0.368]|
| DiffRhythm2 | **七维原始均值唯一reward** |5.116→5.055|−0.061 [−0.179,0.052]|

不能从这张表推导“MOS4+适合AI reward，MOS2–3适合SP”的一般阈值：架构、预算、adapter与β并没有匹配。旧YuE2是PyTorch/PEFT rank4 q/v、β0.01、短semantic-token预算；Diff是Miles DiT attention+MLP rank8、β0.05、60s且只对选中SDE窗口更新；新Muse是ms-swift/TRL主干LoRA rank8、β0.05、512 completion tokens、10.24s MuCodec。SP目标带guard，纯Mean7不带guard，也不是严格reward单变量消融。

## 哪些话不能讲

- 不能说SongEval一定导致削波。YuE2-SongEval3e-4这轮验证平均near-full-scale fraction从约1.346e-6降到1.592e-7；没有复现普遍满幅削波。公司模型音频/权重不可获取，无法证实或归因其现象。MP3听感、采样率、模型/声码器和RL实现仍是不同问题。
- 不能把所有mean上涨都称为质量改善。SongEval是优化指标，Mean7包含训练头，SP是信号代理，都需要独立评价和盲听。
- 不能说Diff Mean7“显著变差”：这个配对区间包含0；准确说法是本配置下未确立验证增益。
- 不能把同条件多个checkpoint/seed当作独立歌曲样本；CI是在30个条件上配对重采样，不是跨训练seed误差条。所有区间未校正多重比较。
- 不能把暂停中的全量test的部分分数当作最终分数。121条件CMI test /192条件WSB尚未在此报告完成独立统一回收，图只呈现完整验证。部分旧评分作业仍为user-held；没有解除暂停、启动新训练或创建定时任务。
- 旧Muse-SongEval/MuseCritic的训练完成日志不等于验证完成。没有完整固定验证的臂不放入效果图，也不拿Muse-SP的结果填充它们。

## 回收与失败记录

- 六个旧YuE2/Diff SP/子项实验重新做了全音频CPU核验，见`legacy-rechecked.json`。新Muse-SP为600条绑定；新Diff Mean7为660条绑定；合计5220条，**含共享baseline与多checkpoint，不是5220首独立歌曲**。最终checkpoint/optimizer/RNG文件已逐个哈希，不把文件存在等价为本次GPU重载测试。
- SongEval三LR重新运行既有formal collector，核验270连续更新、每步两条评分音频及所有0/1/25/50/100/270的30条验证。原始WAV/FLAC和评分文件各自SHA保留。三臂baseline解码后的float32 PCM、采样率和帧数逐条相同；封装文件字节不完全相同，不谎称同一文件SHA。
- Muse-SP从`bolshyC/Muse-0.6b`原始release出发，权重SHA `c810b2a748579e06e960019f89a9cb98701e4ff545e2c760e8ceaa3343ff1b52`，不是旧GRPO adapter。原始权重身份在Oct20的启动凭据也单独保留。
- Muse Instrument181718185、Arrangement181718187：`libpython3.11.so.1.0`缺失，scorer启动前退出，无train.log/optimizer更新。修复方向是显式绑定正确的Python运行库并在GPU作业环境内检查所有服务，而不是改reward。
- YuE2 Mean7/TRL181718819：模型声明与实际gradient-checkpointing接口不匹配，TRL初始化失败，未开始首次更新。需要实现并验证正确的checkpointing路径；不是LoRA参数多寡或模型不能RL的证据。不能把已有PyTorch轮次重命名为TRL结果。

## 图与复现

四张科学问题图，每张三版，数据和置信区间完全一致：A横向差值柱状图、B纵向差值柱状图、C配对端点图（局部数值轴明确标注）。PNG2400×1350，SVG/PDF为矢量。图02两个panel单位不同，不合并数值；图03主比较始终使用原始SongBench Mean7单位。

运行`build_figures.py`读取三份原始回收JSON，固定NumPy seed20261023，配对条件bootstrap10000次，输出`summary.json`、CSV与所有图。图01–04均来自完整30条；失败组没有虚构零分柱子。代码依赖见`requirements-figures.txt`。

`export_audio.py`只在CPU转码已有验证音频：固定valid0/10/20/1/2，两个新实验各baseline/final，共20个MP3，192kbps/48kHz，不做响度归一化。Muse采用最大10.24s预算并保留原生较短输出（例valid0 baseline约9.04s）、Diff60s，不循环或拼接冒充长歌。所有评分FLAC留在私有实验机；公开MP3与原件SHA分开。新增图册提供输入Text/Lyrics、两组新曲线与这些试听对。
