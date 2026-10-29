import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const date = '2026-10-29';
const register = JSON.parse(fs.readFileSync(path.join(root, `data/research-register-${date}.json`)));
const output = path.join(root, `site/amt/research/${date}`);
const check = process.argv.includes('--check');
const escape = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const byId = id => {
  const item = register.items.find(item => item.register_id === id);
  if (!item) throw new Error(`Unknown research reference ${id}`);
  return item;
};
const entryUrl = item => `https://zhanh-he.github.io/RL-paper-reading/${item.catalog === 'amt' ? 'amt/' : ''}?paper=${encodeURIComponent(item.catalog_id)}`;
const ref = id => `<a href="${escape(byId(id).primary_url)}">[${id}]</a>`;
const rows = ids => ids.map(id => {
  const item = byId(id);
  return `<tr><td>${escape(id)} / ${item.review_depth}</td><td><a href="${entryUrl(item)}">${escape(item.title)}</a></td><td>${escape(item.status)}</td><td><a href="${escape(item.primary_url)}">原始来源</a></td></tr>`;
}).join('\n');
const table = ids => `<div class="table-wrap"><table><thead><tr><th>ID / 深度</th><th>文献与目录条目</th><th>时间与状态</th><th>证据</th></tr></thead><tbody>${rows(ids)}</tbody></table></div>`;
const sections = [
  {
    slug: 'audio2score', title: 'Audio-to-Score：前沿与数据条件', ids: register.items.filter(x => x.register_id.startsWith('A')).map(x => x.register_id),
    body: `<h2>先区分任务，再比较结果</h2>
<p>F0、秒级 note MIDI、lead sheet 与完整多声部记谱不是同一种输出。当前公开结果支持流行歌 lead-sheet、钢琴转谱与合成多乐器 A2S 的研究，不支持把其中任何系统称为任意混音完整总谱的统一冠军。</p>
<h2>三个名字不能混用</h2>
<ul><li>Sheet Sage：Donahue 等，ISMIR 2022；melody + chords 的先行工作。${ref('A08')}</li>
<li>SheetSage-A2S / HookKern / SheetSage2Kern：Cummins 等，含 Yaolong Ju；2026-08-06 预印本，官方仓库确认 ACM MM 2026 accepted。会议为 11-10 至 11-14，晚于本报告截止日。${ref('A02')}</li>
<li>SheetSage2：Jiang 等，2026-10-04 的独立预印本；统一 lead-sheet 多属性预测，提供 ABC、MIDI 与事件输出；不是 SATB 总谱系统。${ref('A01')}</li></ul>
<h2>前沿结果的适用范围</h2>
<p>SheetSage2 报告在八个 benchmark 集合的十五个 task–metric 对中领先所列既有系统的十二项；其 vocal-melody F1 忽略八度和 offset，因此不能据此推断绝对音高、时值或 VST 回放已可靠。HookKern 的 SER、TUTTI 的 MV2H、Rubato 的 OMR-NED 与 MusicTimeA2S 的整曲协议不能直接混排。${ref('A01')} ${ref('A02')} ${ref('A04')} ${ref('A03')} ${ref('A05')}</p>
<p>MuScriptor 的主要论文评价是 audio-to-MIDI；当前工具的谱面导出是另一层能力。A Dual Evaluation 则提醒我们把记谱正确性、可读性和人类感知分开测量。${ref('A10')} ${ref('A09')}</p>
<h2>数据支持度</h2>
<div class="table-wrap"><table><thead><tr><th>资源</th><th>已确认的标签/配对</th><th>不能据此假定</th></tr></thead><tbody>
<tr><td>SheetSage-A2S</td><td>pop lead-sheet melody/chord kern；申请制数据和特征版本</td><td>无条件可重分发真实音频、完整编配或 SATB labels</td></tr>
<tr><td>TuttiCorpus</td><td>363,610 合成 audio-score pairs；多乐器 ABC</td><td>真实商业混音表现；全镜像已由我们下载验收</td></tr>
<tr><td>Gangqinpu</td><td>18,786 钢琴 score-performance pairs；AMT + refinement MIDI</td><td>硬件采集的 performance-MIDI ground truth</td></tr>
<tr><td>MulTTiPop</td><td>572 clips、约 3.5 h；对齐 multitrack MIDI，作者建议 evaluation-only</td><td>独立真实 stems、适合直接作训练集</td></tr>
</tbody></table></div>
<p>以上资源与入口分别见 ${ref('A02')} ${ref('A04')} ${ref('A11')} ${ref('A12')}。需要 score、MIDI、stems、F0、clean restoration target 时，应逐列核对，而不是用“有标注”概括全部条件。</p>
<h2>覆盖范围与未验证项</h2><p>本页收录的 program-only 论文不参与数值 SOTA 排名；预印本没有录用证据时仍标预印本。未运行模型、未申请 gated 数据，也未验证全量数据可下载性。</p>`
  },
  {
    slug: 'generation', title: 'Audio-to-Score 与音乐生成：证据和评估接口', ids: register.items.filter(x => x.register_id.startsWith('G') || ['J02','J03','J04','J05'].includes(x.register_id)).map(x => x.register_id),
    body: `<h2>符号中间表示已经有多条先例</h2>
<p>符号规划再渲染不是空白方向：SymPAC、BACH、Text2Score、YuE2 分别覆盖提示/约束、bar-level notation、文本转谱及统一 symbolic/audio generation。需要逐项区分符号监督、生成时条件、可编辑输出和实际音频控制。${ref('G05')} ${ref('G03')} ${ref('G04')} ${ref('G02')}</p>
<h2>MIDI-REPA 实际做了什么</h2>
<p>MIDI-REPA 在训练时使用 AMT piano-roll 表示对齐生成网络的隐层，不是一个推理时生成完整乐谱的 A2S 模型。其短时 instrumental 实验不能自动外推到长歌曲、歌声或声部保持；论文也呈现 teacher 噪声与不同表示/指标之间的取舍。${ref('G01')}</p>
<p><a href="https://github.com/vaclisinc/MIDI-REPA">训练代码</a>、<a href="https://github.com/vaclisinc/AMTFlow">AMTFlow</a> 和 <a href="https://huggingface.co/datasets/vaclis/JamendoMIDI">JamendoMIDI</a> 是不同资源。JamendoMIDI 的 51,077 instrumental tracks 使用 pseudo-MIDI，不应作为真实 note ground truth。模型、代码、原始音频和数据集许可证也不能相互代替。</p>
<h2>音乐 CoT 要核对接口，而不只看名字</h2>
<p>显式小节计划、ABC、note tokens、自然语言推理和机器可检查的编辑日志是不同中间表示。Music Flamingo 与 MuNo-SP 是音乐理解方向；理解 benchmark 成绩不等于生成控制的因果收益。${ref('G07')} ${ref('G08')} ${ref('G10')}</p>
<h2>可用于比较系统的评价维度</h2>
<p>以下是方法比较建议，不是已完成实验或新颖性结论：分别报告目标编辑成功、未编辑角色保持、绝对音高/offset、小节定位、音频质量、失败覆盖率与推理预算。若用自动转录器评分，应加入独立系统或人工核对，防止同一 teacher 既提供监督又决定成败。</p>
<p>合成精确符号配对适合定位问题；真实数据适合检查域差异。两者需要按作品/编配去重，记录 renderer、teacher、模型版本和许可证。公开文献并不能证明任何具体内部选题没有与他人未公开工作重叠。</p>
<h2>相关作者的公开发表状态</h2>
<p>Yaolong Ju 参与 SheetSage-A2S、PawCT、SCALE/SheetSage-MSA、Diff-Symbo、CrossMuSim 与 Sing2Song。PawCT 的“submitted to ICASSP 2027”不是 accepted；MusicDETR 不应因推荐关系被写成其署名成果。这里只列公开题录，不公开组内判断或未发表设计。${ref('A02')} ${ref('J01')} ${ref('J02')} ${ref('J03')} ${ref('J04')} ${ref('J05')}</p>`
  },
  {
    slug: 'midi2score', title: 'MIDI-to-Score 与符号旋律识别', ids: register.items.filter(x => x.register_id.startsWith('S') || x.register_id === 'J01').map(x => x.register_id),
    body: `<h2>MIDI 存在，不等于已经有 score</h2>
<p>Performance MIDI 的秒级时序需要节拍/小节、量化、拼写、voice/staff 等记谱推断；quantized score MIDI 则已经给出了部分这些信息。简单导入 notation 软件与学习 performance-to-score 不是同一项任务。</p>
<h2>按输入选择 baseline</h2>
<ul><li>演奏 MIDI 转谱：DuET、MIDI2ScoreTransformer、PM2S。${ref('S01')} ${ref('S02')} ${ref('S03')}</li>
<li>已量化 MIDI 的完整记谱：Score Transformer；其 ISMIR 2021 记录是 Late Breaking Demo，不应标作正式 full paper。${ref('S04')}</li>
<li>voice/staff engraving：Cluster and Separate、MusGConv；这些输出不等于 lead/accompaniment 语义角色。${ref('S06')} ${ref('S07')}</li></ul>
<p>DuET 的 frame 分支处理 MIDI 表示，不是 raw audio。它对输入音符逐一赋记谱属性，复制输入 pitch，并不通过增删音符或 ornament reduction 修复任意 AMT 错误。官方 audio 接口串联 Transkun 与 DuET，需区分端到端错误和 MIDI-to-score 错误。${ref('S01')}</p>
<h2>Lead / Accompaniment 的三种情况</h2>
<ul><li>已分轨 MIDI：Robust Melody Track Identification 是直接参照，带 D-Full 数据资源。${ref('S08')}</li>
<li>旋律和伴奏在同轨混合：MidiBERT-Piano 的 CP 分支含 melody-extraction 工具；CNN melody-line 与 SMC 2019 方法提供不同建模参照。${ref('S09')} ${ref('S10')} ${ref('S11')}</li>
<li>voice / staff 预测：属于记谱结构；不能固定把最高音、右手、SATB soprano 或某个 MIDI program 当作主旋律。${ref('S06')} ${ref('S07')}</li></ul>
<h2>数据与评估注意点</h2>
<p><a href="https://github.com/music-x-lab/POP909-Dataset">POP909</a> 的 MELODY / BRIDGE / PIANO 角色有监督价值，但不是原歌曲所有音源的逐音真值。<a href="https://github.com/CPJKU/asap-dataset">ASAP</a> 提供钢琴 score/performance 配对；<a href="https://github.com/pnlong/PDMX">PDMX</a> 适合符号学习与自行渲染，不天然包含真实演奏配对。</p>
<p>Lead-role 测试应允许跨轨交接、多主旋律及无主旋律片段，分别测 note、track 和 segment。合唱 part assignment 是另一层问题；PawCT 可以作为 SATB 扩展参照，但不等于通用主旋律识别。${ref('J01')}</p>`
  }
];

const navigation = `<nav><a href="../../">AMT catalog</a><a href="../../../">MusicGen / RL catalog</a><a href="index.html">专题总览</a>${sections.map(s => `<a href="${s.slug}.html">${escape(s.title.split('：')[0])}</a>`).join('')}</nav>`;
const shell = (title, body) => `<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>${escape(title)} | AMT Research</title>
<style>body{margin:0;background:#fff;color:#222;font:16px/1.7 system-ui,sans-serif;letter-spacing:0}main{max-width:1080px;margin:auto;padding:24px}nav{display:flex;flex-wrap:wrap;gap:8px 20px;border-bottom:1px solid #ddd;padding-bottom:16px}a{color:#186146;text-underline-offset:3px}h1{font-size:28px;line-height:1.35;margin:28px 0 12px}h2{font-size:21px;margin-top:28px}p,li,td{overflow-wrap:anywhere}.meta{color:#555;font-size:14px}.table-wrap{max-width:100%;overflow:auto}table{width:100%;border-collapse:collapse;font-size:14px}th,td{border-bottom:1px solid #ddd;text-align:left;vertical-align:top;padding:10px}th{background:#f2f5f3}th:first-child{min-width:70px}footer{margin-top:30px;border-top:1px solid #ddd;padding-top:14px;font-size:14px;color:#555}@media(max-width:600px){main{padding:16px}h1{font-size:24px}table{min-width:650px}}</style></head>
<body><main>${navigation}<h1>${escape(title)}</h1><p class="meta">核查截止 ${date}（Australia/Perth）；arXiv 提交日按 UTC。公开来源整理，不是模型复现。T = 核对原文关键方法/结果/限制；A = 摘要或官方资源；M = 题录/program。</p>${body}<footer>共享条目只保留一个主目录；点击题名打开稳定 catalog deep link。缺少评分、单位或资源链接表示未登记/待核验，不能当作零分或资源不存在。私人研究笔记与合作讨论不在此公开。</footer></main></body></html>\n`;
const pages = new Map(sections.map(s => [`${s.slug}.html`, shell(s.title, `${s.body}<h2>文献与发表时间</h2>${table(s.ids)}`)]));
pages.set('index.html', shell('AMT / Score / Generation 文献导航', `<p>三条阅读线：先明确转录的输出层级，再判断符号表示能否服务生成，最后选择 MIDI 转谱和角色识别工具。这里链接 ${register.items.length} 条去重后的共享记录；完整私人分析保留在 Obsidian。</p><ul>${sections.map(s => `<li><a href="${s.slug}.html">${escape(s.title)}</a></li>`).join('')}</ul><h2>使用边界</h2><p>不同任务、数据版本和指标不能构成一个混排 SOTA 榜单。尚未完成模型实测、全量数据下载验收或所有预印本最终 venue 核查。仅题录核查的条目明确标 M，不据此推荐性能。</p><h2>全部来源</h2>${table(register.items.map(x => x.register_id))}`));
for (const [name, content] of pages) {
  const filename = path.join(output, name);
  if (check) {
    if (!fs.existsSync(filename) || fs.readFileSync(filename, 'utf8') !== content) throw new Error(`Stale research page: ${name}`);
  } else {
    fs.mkdirSync(output, {recursive:true});
    fs.writeFileSync(filename, content);
  }
}
console.log(`Research guide: ${pages.size} pages, ${register.items.length} references (${check ? 'checked' : 'built'})`);
