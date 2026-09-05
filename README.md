# RL Paper Reading
[Open interactive catalog](https://zhanh-he.github.io/RL-paper-reading/) · [Update guide](EDITING.md)

## How To Use

1. 在 [interactive catalog](https://zhanh-he.github.io/RL-paper-reading/) 搜索、筛选和排序文献；点击标题查看摘要及 paper/code/demo 链接。
2. 需要新增论文、修改评分或纠正内容时，编辑自己的 request 文件并提交：
   - [Zhanh request](https://github.com/zhanh-he/RL-paper-reading/edit/main/Update_request_zhanh.txt)
   - [Felix request](https://github.com/zhanh-he/RL-paper-reading/edit/main/Update_request_felix.txt)
   - [Hanyu request](https://github.com/zhanh-he/RL-paper-reading/edit/main/Update_request_hanyu.txt)
3. 希望更改的Request会被提交给gpt，在gpt完成后会带时间戳归档到 [`Done_UPD_request/`](Done_UPD_request/)，原位置会留下新的空白模板供下次使用。

也可以直接修改 [`data/literature.csv`](data/literature.csv)，但不要手改 `site/data/literature.json`；它们由 GitHub Actions 自动生成。

## Labels
- Domain：`MusicGen / MusicEval / MIR / AudioGen / SpeechEnhance / AudioLLM / LLM / CV / ML / SourceSep / Multimodal / Other`
- Workstream：`Reward / RL / Reward-n-RL / Other`
- Felix 与 Zhanh 的星级
