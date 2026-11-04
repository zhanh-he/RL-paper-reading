const root = document.querySelector('#naturalness-content');
const finite = (value) => typeof value === 'number' && Number.isFinite(value);
const fmt = (value, digits = 3) => finite(value) ? value.toFixed(digits) : '未运行';
const asText = (value) => value == null ? '' : typeof value === 'string' ? value : Array.isArray(value) ? value.map(asText).filter(Boolean).join('；') : typeof value === 'object' ? asText(value.text || value.description || value.result || value.note || value.conclusion || value.status) : String(value);
const first = (obj, keys, fallback = '') => keys.map((key) => obj?.[key]).find((value) => value != null && value !== '') ?? fallback;
function el(tag, text, className) { const node = document.createElement(tag); if (text != null) node.textContent = text; if (className) node.className = className; return node; }
function safeUrl(value) { if (typeof value !== 'string') return null; try { const u = new URL(value, location.href); return ['http:', 'https:'].includes(u.protocol) && !/^\/(Users|home|storage|media)\//.test(value) ? u.href : null; } catch { return null; } }
function link(title, url) { const a = el('a', title); const href = safeUrl(url); if (href) { a.href = href; a.target = '_blank'; a.rel = 'noopener noreferrer'; } return a; }
function button(title, callback) { const b = el('button', title, 'nat-button'); b.type = 'button'; b.addEventListener('click', callback); return b; }
function field(title, control) { const label = el('label', title); label.append(control); return label; }
function select(options, label) { const s = el('select'); s.setAttribute('aria-label', label); for (const [value, title] of options) { const o = el('option', title); o.value = value; s.append(o); } return s; }
function checkbox(title, checked) { const input = el('input'); input.type = 'checkbox'; input.checked = checked; const label = el('label', null, 'nat-check'); label.append(input, document.createTextNode(title)); return { input, label }; }
function details(title, children = []) { const d = el('details', null, 'nat-block'); d.append(el('summary', title), ...children); return d; }
function list(values) { const ul = el('ul', null, 'nat-inline-list'); for (const value of values || []) if (asText(value)) ul.append(el('li', asText(value))); return ul; }
function table(headers, rows, className = '') { const wrap = el('div', null, 'nat-table-wrap'); wrap.tabIndex = 0; const t = el('table', null, className); const head = el('thead'), tr = el('tr'); headers.forEach((h) => { const th = el('th', h); th.scope = 'col'; tr.append(th); }); head.append(tr); const body = el('tbody'); rows.forEach((values) => { const r = el('tr'); values.forEach((value) => { const td = el('td'); if (value instanceof Node) td.append(value); else td.textContent = asText(value) || '—'; r.append(td); }); body.append(r); }); t.append(head, body); wrap.append(t); return wrap; }
function download(name, content, type) { const url = URL.createObjectURL(new Blob([content], { type })); const a = el('a'); a.href = url; a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); }
const ratingFields = [['artificialness', '不自然人工感', '1 很自然', '5 缺陷很强'], ['quality', '整体音质', '1 很差', '5 很好'], ['musicality', '音乐性', '1 很差', '5 很好']];
const preferenceFields = [['naturalness', '自然度更好'], ['quality', '音质更好'], ['musicality', '音乐性更好']];
const metricDefs = [
  ['audiobox.PQ', 'AudioBox PQ ↑', '高分偏生产质量，不是电音概率'],
  ['audiobox.PC', 'AudioBox PC', '制作复杂度，不是乐器数量'],
  ['audiobox.CE', 'AudioBox CE', '内容愉悦度，不等于音乐性评分'],
  ['audiobox.CU', 'AudioBox CU', '内容实用性，不等于音乐性评分'],
  ['songeval.Naturalness', 'SongEval 人声分句 ↑', '官方Naturalness头主要衡量呼吸与乐句，非全混音伪影检测'],
  ['songeval.Musicality', 'SongEval 音乐性 ↑', 'SongEval已训练音乐性头'],
  ['pam.score', 'PAM ↑', '文本提示对比得分，不是校准概率'],
  ['pam.artifact_score', 'PAM 自拟提示诊断', '自拟伪影提示实验；非官方PAM，失败时不补分'],
  ['signal.rms_dbfs', 'RMS dBFS', '音量诊断'],
  ['signal.near_full_scale_fraction', '近满幅比例 ↓', '样本接近满幅；不等价于可听失真'],
  ['signal.flatness', '频谱平坦度', '噪声/音调结构诊断；无通用优劣方向'],
  ['signal.envelope_modulation_20_100_share', '20–100 Hz 调制占比', '调制诊断；颤音、粗糙音色也会升高'],
];
const getMetric = (sample, path) => path.split('.').reduce((o, key) => o?.[key], sample);
function category(s) {
  const k = String(s.kind || '').toLowerCase();
  if (/construct|control/.test(k) && !/perturb|transform/.test(k)) return 'synthetic';
  if (/reference/.test(k) && /emma|ace/i.test(s.model || '')) return 'synthetic_voice';
  if (/transform|perturb|attack/.test(k) || (s.transform && !['original', 'none', 'identity'].includes(typeof s.transform === 'string' ? s.transform : s.transform.name))) return 'transform';
  if (/record|natural|acoustic|reference/.test(k)) return 'recorded';
  if (/synth|silence/.test(k)) return 'synthetic';
  return 'generated';
}
const categoryLabels = { generated: '真实生成输出', transform: '受控变换', recorded: '录制参考', synthetic: '构造信号 / 合成器', synthetic_voice: '合成歌声参考' };

async function load() {
  root.replaceChildren(el('p', '正在读取实验数据与试听清单…', 'nat-note'));
  try {
    const response = await fetch('./data/naturalness-audit.json', { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    if (!Array.isArray(data.samples) || !data.samples.length || data.samples.some((s) => !s.id || !safeUrl(s.audio))) throw new Error('样本清单不完整');
    render(data);
  } catch (error) {
    root.replaceChildren(el('p', `实验清单暂时无法读取（${error.message}）。不显示替代分数。`, 'nat-note'), button('重新读取', load));
  }
}

function render(data) {
  root.replaceChildren();
  const samples = data.samples;
  const byId = new Map(samples.map((s) => [s.id, s]));
  if (byId.size !== samples.length) throw new Error('样本ID重复');
  const meta = data.metadata || {};
  const version = String(meta.version || meta.dataset_version || '20261104-v1');
  const storageKey = `rl-lab-naturalness:${version}`;
  let saved = { ratings: {}, pairs: {}, rater: '' };
  try { const old = JSON.parse(localStorage.getItem(storageKey)); if (old?.ratings && old?.pairs) saved = old; } catch { /* a fresh blank form is intentional */ }
  const status = el('p', '', 'nat-status'); status.setAttribute('role', 'status');
  const persist = () => { try { localStorage.setItem(storageKey, JSON.stringify(saved)); status.textContent = '已保存于本机浏览器；不会自动上传。'; } catch { status.textContent = '浏览器未允许本机保存；请导出JSON保留评分。'; } };
  const anonymousName = (s) => `样本 ${String(samples.indexOf(s) + 1).padStart(2, '0')}`;
  let blind = true, currentA = samples[0].id, currentB = '', filter = 'all';
  const matching = (s) => byId.get(s.matched_comparator_id) || samples.find((x) => x.id !== s.id && ((s.pair_group && x.pair_group === s.pair_group) || x.base_id === s.id || s.base_id === x.id));
  currentB = (matching(samples[0]) || samples[1] || samples[0]).id;
  const name = (s) => blind ? anonymousName(s) : s.label || s.id;

  const summary = typeof data.summary === 'string' ? { conclusion: data.summary } : data.summary || {};
  const intro = el('div', null, 'nat-summary');
  intro.append(el('p', asText(first(summary, ['conclusion', 'headline', 'recommendation'], '先完成受控审计与人工评分，再决定是否作为RL目标。')), 'nat-conclusion'));
  intro.append(el('p', asText(summary.evidence) || `${samples.length}条固定片段；人工标签为${meta.human_labels_count ?? 0}，本轮没有新RL训练。`, 'nat-note'));
  const notes = [summary.scope, ...(Array.isArray(summary.limitations) ? summary.limitations : []), ...(Array.isArray(meta.limitations) ? meta.limitations : [])].filter(Boolean);
  if (meta.background_document?.status === 'missing_not_read') intro.append(el('p', '背景Word文件未找到，尚未阅读；目标架构仍待核对。', 'nat-note'));
  root.append(intro);

  const listening = el('section', null, 'nat-block'); listening.append(el('h3', '先听同一片段，再分别判断'));
  const tools = el('div', null, 'nat-toolbar');
  const filterSelect = select([['all', '全部样本'], ...Object.entries(categoryLabels)], '样本类型');
  const blindControl = checkbox('隐藏来源名称与自动分数', true);
  const syncControl = checkbox('A/B切换时保持播放进度', true);
  tools.append(field('样本类型', filterSelect), blindControl.label, syncControl.label); listening.append(tools);
  listening.append(el('p', '先以匿名编号试听；正常电子乐器、合成器和明亮音色本身不算缺陷。选择样本不自动播放，A/B可保持进度切换。用稳定音量，分别判断人工感、音质和音乐性。', 'nat-note'));
  const players = el('div', null, 'nat-players'); listening.append(players);
  const slots = {};
  function buildPlayer(side) {
    const section = el('article', null, 'nat-player');
    const sampleSelect = select([], `${side}样本`); const heading = el('h4');
    const audio = el('audio'); audio.controls = true; audio.preload = 'metadata'; audio.setAttribute('aria-label', `${side}试听播放器`);
    const audioStatus = el('p', '', 'nat-status');
    audio.addEventListener('play', () => document.querySelectorAll('audio').forEach((other) => { if (other !== audio) other.pause(); }));
    audio.addEventListener('error', () => { audioStatus.textContent = '音频加载失败，请检查网络后重选样本或重新读取数据。'; });
    const scores = el('div', null, 'nat-scores');
    const scoreControls = el('div', null, 'nat-rating-grid');
    const ratings = {};
    for (const [key, title, low, high] of ratingFields) {
      const control = select([['', '待评价'], ['1', low], ['2', '2'], ['3', '3'], ['4', '4'], ['5', high]], `${side} ${title}`);
      control.addEventListener('change', () => saveRating(side)); ratings[key] = control; scoreControls.append(field(title, control));
    }
    const note = el('textarea'); note.rows = 2; note.maxLength = 4000; note.placeholder = '例如：持续金属蜂鸣／瞬态模糊／属于正常合成器；不确定也请记录。'; note.addEventListener('change', () => saveRating(side));
    const provenance = el('details'); provenance.append(el('summary', '片段来源与全部诊断')); const provenanceBody = el('div'); provenance.append(provenanceBody);
    const switchButton = button(`试听 ${side}`, () => {
      const other = slots[side === 'A' ? 'B' : 'A'].audio;
      const time = syncControl.input.checked && Number.isFinite(other.currentTime) ? other.currentTime : audio.currentTime || 0;
      other.pause();
      const play = () => { audio.currentTime = Math.max(0, Math.min(time, finite(audio.duration) ? Math.max(0, audio.duration - .01) : time)); audio.play().catch(() => { audioStatus.textContent = '请直接点击播放器播放。'; }); };
      if (audio.readyState >= 1) play(); else { audio.addEventListener('loadedmetadata', play, { once: true }); audio.load(); }
    });
    section.append(field(`${side} · 选择样本`, sampleSelect), heading, audio, switchButton, audioStatus, scores, scoreControls, field('听感备注', note), provenance);
    players.append(section);
    slots[side] = { section, sampleSelect, heading, audio, audioStatus, scores, ratings, note, provenance, provenanceBody };
    sampleSelect.addEventListener('change', () => { if (side === 'A') currentA = sampleSelect.value; else currentB = sampleSelect.value; updatePlayer(side); updatePair(); });
  }
  buildPlayer('A'); buildPlayer('B');
  function saveRating(side) {
    const s = byId.get(side === 'A' ? currentA : currentB), slot = slots[side];
    saved.ratings[s.id] = { sample_id: s.id, artificialness: slot.ratings.artificialness.value ? Number(slot.ratings.artificialness.value) : null, quality: slot.ratings.quality.value ? Number(slot.ratings.quality.value) : null, musicality: slot.ratings.musicality.value ? Number(slot.ratings.musicality.value) : null, note: slot.note.value.trim(), blind, updated_at: new Date().toISOString() };
    persist(); if (currentA === currentB) updatePlayer(side === 'A' ? 'B' : 'A');
  }
  function updateOptions() {
    for (const side of ['A', 'B']) {
      const control = slots[side].sampleSelect, selected = side === 'A' ? currentA : currentB;
      const visible = samples.filter((s) => filter === 'all' || category(s) === filter || s.id === selected);
      control.replaceChildren(...visible.map((s) => { const o = el('option', name(s)); o.value = s.id; return o; })); control.value = selected;
    }
  }
  function updatePlayer(side) {
    const s = byId.get(side === 'A' ? currentA : currentB), slot = slots[side], rating = saved.ratings[s.id] || {};
    slot.heading.textContent = `${side} · ${name(s)} · ${finite(s.duration_s) ? s.duration_s.toFixed(1) + '秒' : '固定片段'}`;
    if (slot.audio.dataset.sample !== s.id) { slot.audio.pause(); slot.audio.src = safeUrl(s.audio); slot.audio.dataset.sample = s.id; slot.audioStatus.textContent = ''; }
    for (const [key] of ratingFields) slot.ratings[key].value = rating[key] == null ? '' : String(rating[key]);
    slot.note.value = rating.note || ''; slot.scores.hidden = blind; slot.provenance.hidden = blind;
    slot.scores.replaceChildren(...metricDefs.slice(0, 8).map(([path, title]) => el('span', `${title}: ${fmt(getMetric(s, path))}`)));
    const dl = el('dl', null, 'nat-meta');
    for (const [title, value] of [['类型', categoryLabels[category(s)]], ['模型', s.model], ['阶段', s.checkpoint], ['条件', s.input_id], ['作者署名', s.credit], ['选择原因', s.selection_reason], ['变换', asText(s.transform)], ['来源原件SHA', s.source_original_sha256], ['评分片段SHA', s.excerpt_sha256], ['试听MP3 SHA', s.mp3_sha256], ['使用依据', s.rights]]) if (value) dl.append(el('dt', title), el('dd', asText(value)));
    const signal = Object.entries(s.signal || {}).filter(([, value]) => finite(value)).map(([key, value]) => `${key}: ${fmt(value, 5)}`);
    slot.provenanceBody.replaceChildren(dl, ...(s.source_url ? [link('已发布来源', s.source_url)] : []), ...(s.license_url ? [link('素材许可', s.license_url)] : []), ...(signal.length ? [list(signal)] : []));
    if (s.audiobox?.error || s.pam?.error) slot.provenanceBody.append(el('p', `方法失败：${asText(s.audiobox?.error || s.pam?.error)}`, 'nat-note'));
  }

  const pairBlock = el('section', null, 'nat-block'); pairBlock.append(el('h3', 'A/B 两两比较'));
  const pairNote = el('p', '', 'nat-note'); const pairGrid = el('div', null, 'nat-pair-grid'); const prefs = {};
  for (const [key, title] of preferenceFields) {
    const control = select([['', '待评价'], ['A', 'A 更好'], ['B', 'B 更好'], ['tie', '相当'], ['uncertain', '无法判断']], title);
    control.addEventListener('change', savePair); prefs[key] = control; pairGrid.append(field(title, control));
  }
  const pairComment = el('textarea'); pairComment.rows = 2; pairComment.maxLength = 4000; pairComment.addEventListener('change', savePair);
  const pairKey = () => JSON.stringify([currentA, currentB]);
  function savePair() { saved.pairs[pairKey()] = { a: currentA, b: currentB, ...Object.fromEntries(preferenceFields.map(([key]) => [key, prefs[key].value || null])), note: pairComment.value.trim(), blind, updated_at: new Date().toISOString() }; persist(); }
  function updatePair() {
    const a = byId.get(currentA), b = byId.get(currentB), sameCondition = a.input_id && a.input_id === b.input_id;
    pairNote.textContent = currentA === currentB ? '当前A、B是同一个样本。' : sameCondition || a.base_id === b.id || b.base_id === a.id ? '当前为相同输入／同源变换对照。生成前后片段的音乐内容仍可能变化；差异不能自动归因于伪影。' : '当前内容或条件不同；可记录偏好，但不能把内容偏好解释成电音缺陷识别。';
    const record = saved.pairs[pairKey()] || {}; for (const [key] of preferenceFields) prefs[key].value = record[key] || ''; pairComment.value = record.note || '';
  }
  pairBlock.append(pairNote, button('B 选择 A 的同源对照', () => { const match = matching(byId.get(currentA)); if (match) { currentB = match.id; updateOptions(); updatePlayer('B'); updatePair(); } else status.textContent = '当前样本没有预设同源对照，可手动选择B。'; }), pairGrid, field('比较备注', pairComment));
  listening.append(pairBlock);
  const rater = el('input'); rater.type = 'text'; rater.maxLength = 80; rater.value = saved.rater || ''; rater.placeholder = '可选，仅保存在本机'; rater.addEventListener('change', () => { saved.rater = rater.value.trim(); persist(); });
  const exportBar = el('div', null, 'nat-export');
  const exportData = () => ({ schema: 'rl-lab-naturalness-labels-v1', dataset_version: version, rater: saved.rater || '', exported_at: new Date().toISOString(), scales: { artificialness: '1 natural to 5 strong unwanted artificial artifacts', quality: '1 poor to 5 excellent', musicality: '1 poor to 5 excellent' }, ratings: Object.values(saved.ratings), pairs: Object.values(saved.pairs) });
  exportBar.append(button('导出评分 JSON', () => download(`naturalness-labels-${version}.json`, JSON.stringify(exportData(), null, 2), 'application/json')));
  exportBar.append(button('导出评分 CSV', () => {
    const rows = [['record_type', 'dataset_version', 'rater', 'sample_id', 'a', 'b', 'artificialness', 'quality', 'musicality', 'naturalness_preference', 'quality_preference', 'musicality_preference', 'note', 'blind', 'updated_at']];
    for (const s of samples) { const r = saved.ratings[s.id] || {}; rows.push(['sample', version, saved.rater, s.id, '', '', r.artificialness, r.quality, r.musicality, '', '', '', r.note, r.blind, r.updated_at]); }
    for (const r of Object.values(saved.pairs)) rows.push(['pair', version, saved.rater, '', r.a, r.b, '', '', '', r.naturalness, r.quality, r.musicality, r.note, r.blind, r.updated_at]);
    const escape = (v) => { let s = v == null ? '' : String(v); if (/^[=+@\-\t\r]/.test(s)) s = "'" + s; return '"' + s.replaceAll('"', '""') + '"'; };
    download(`naturalness-labels-${version}.csv`, '\uFEFF' + rows.map((r) => r.map(escape).join(',')).join('\r\n'), 'text/csv;charset=utf-8');
  }));
  const importFile = el('input'); importFile.type = 'file'; importFile.accept = 'application/json,.json'; importFile.hidden = true;
  exportBar.append(button('导入评分 JSON', () => importFile.click()), importFile);
  importFile.addEventListener('change', async () => {
    try {
      const file = importFile.files[0]; if (!file) return; if (file.size > 2e6) throw new Error('文件过大'); const incoming = JSON.parse(await file.text());
      if (incoming.schema !== 'rl-lab-naturalness-labels-v1' || incoming.dataset_version !== version) throw new Error('数据版本或评分格式不匹配');
      if (!Array.isArray(incoming.ratings) || !Array.isArray(incoming.pairs)) throw new Error('评分数组缺失');
      const ratings = {}, pairs = {};
      for (const r of incoming.ratings) { if (!byId.has(r.sample_id) || ratingFields.some(([key]) => r[key] != null && (!Number.isInteger(r[key]) || r[key] < 1 || r[key] > 5))) throw new Error('样本ID或评分范围无效'); ratings[r.sample_id] = { sample_id: r.sample_id, ...Object.fromEntries(ratingFields.map(([key]) => [key, r[key] ?? null])), note: String(r.note || '').slice(0, 4000), blind: !!r.blind, updated_at: r.updated_at || new Date().toISOString() }; }
      for (const r of incoming.pairs) { if (!byId.has(r.a) || !byId.has(r.b) || preferenceFields.some(([key]) => ![null, undefined, '', 'A', 'B', 'tie', 'uncertain'].includes(r[key]))) throw new Error('比较ID或偏好值无效'); pairs[JSON.stringify([r.a, r.b])] = { a: r.a, b: r.b, ...Object.fromEntries(preferenceFields.map(([key]) => [key, r[key] || null])), note: String(r.note || '').slice(0, 4000), blind: !!r.blind, updated_at: r.updated_at || new Date().toISOString() }; }
      saved = { ratings: { ...saved.ratings, ...ratings }, pairs: { ...saved.pairs, ...pairs }, rater: String(incoming.rater || saved.rater || '').slice(0, 80) }; rater.value = saved.rater; persist(); updatePlayer('A'); updatePlayer('B'); updatePair(); status.textContent = `已导入 ${Object.keys(ratings).length} 条单样本、${Object.keys(pairs).length} 条比较；同ID评分采用导入值。`;
    } catch (error) { status.textContent = `未导入：${error.message}。现有评分已保留。`; } finally { importFile.value = ''; }
  });
  listening.append(field('评分者别名', rater), exportBar, status); root.append(listening);

  if (notes.length) root.append(details('范围与未确认项', [list(notes), el('p', '当前匿名界面用于本机评分辅助，尚未完成预注册的正式盲测。', 'nat-note')]));
  const credits = [...new Map(samples.filter(s => s.credit && s.source_url).map(s => [s.source_url, s])).values()];
  if (credits.length) root.append(details('试听素材署名与许可（完成匿名听评后展开）', credits.map(s => {
    const p = el('p'); p.append(document.createTextNode(`${s.credit} — `), link(s.input_id === 'librosa example vibeace' ? 'Vibe Ace' : 'Let’s Go Fishin’', s.source_url), document.createTextNode(' · '), link('CC-BY-3.0', s.license_url), document.createTextNode('。经 librosa 取得 OGG 分发源，截取10秒、部分片段作受控变换，再编码 MP3；原作者未背书本研究。')); return p;
  })));

  const scoreArea = details('自动分数与排序（取消隐藏后可见）'); const scoreContent = el('div'); scoreArea.append(scoreContent); root.append(scoreArea);
  const sortControl = select([['original', '原始固定顺序'], ...metricDefs.map(([key, title]) => [key, title])], '排序指标');
  const direction = select([['desc', '数值从高到低'], ['asc', '数值从低到高']], '排序方向');
  const sortTools = el('div', null, 'nat-toolbar'); sortTools.append(field('排序指标', sortControl), field('顺序', direction));
  const rank = el('div'); scoreContent.append(el('p', '同一列才可比较。↑是该方法分数方向，不代表本任务已验证自然度。SP指标不用于通用好坏排名；缺值保持未运行。', 'nat-note'), sortTools, rank);
  function drawRanking() {
    scoreContent.hidden = blind; scoreArea.querySelector('summary').textContent = blind ? '自动分数与排序 · 已隐藏，先完成试听' : '自动分数与排序';
    const rows = samples.filter((s) => filter === 'all' || category(s) === filter);
    if (sortControl.value !== 'original') rows.sort((a, b) => { const av = getMetric(a, sortControl.value), bv = getMetric(b, sortControl.value); if (!finite(av)) return finite(bv) ? 1 : 0; if (!finite(bv)) return -1; return (direction.value === 'asc' ? 1 : -1) * (av - bv); });
    rank.replaceChildren(table(['样本 · 点击载入 A', ...metricDefs.map(([, title]) => title)], rows.map((s) => { const open = el('button', s.label || s.id, 'nat-row-button'); open.type = 'button'; open.addEventListener('click', () => { currentA = s.id; updateOptions(); updatePlayer('A'); updatePair(); listening.scrollIntoView({ behavior: 'smooth', block: 'start' }); }); return [open, ...metricDefs.map(([key]) => fmt(getMetric(s, key)))]; }), 'nat-rank-table'));
  }
  sortControl.addEventListener('change', drawRanking); direction.addEventListener('change', drawRanking);
  blindControl.input.addEventListener('change', () => { blind = blindControl.input.checked; updateOptions(); updatePlayer('A'); updatePlayer('B'); drawRanking(); });
  filterSelect.addEventListener('change', () => { filter = filterSelect.value; const visible = samples.filter((s) => filter === 'all' || category(s) === filter); if (visible.length && !visible.some((s) => s.id === currentA)) currentA = visible[0].id; updateOptions(); updatePlayer('A'); updatePlayer('B'); updatePair(); drawRanking(); });

  const methods = Array.isArray(data.methods) ? data.methods : [];
  root.append(details('候选方法 · 测量对象、域与失败风险', [table(['方法', '实际测量／适用域', '运行状态', '局限与误伤'], methods.map((m) => [m.url ? link(asText(first(m, ['name', 'title', 'id'])), m.url) : asText(first(m, ['name', 'title', 'id'])), asText(first(m, ['measures', 'measurement', 'description', 'domain'])), asText(first(m, ['status', 'run_status'], '未运行')), asText(first(m, ['failure_modes', 'limitations', 'risk', 'note', 'reason']))]), 'nat-method-table')]));
  const comparisons = Array.isArray(data.comparisons) ? data.comparisons : [];
  if (comparisons.length) {
    const figure = el('figure', null, 'nat-response-figure');
    const image = el('img'); image.src = './visuals/naturalness-response.png'; image.loading = 'lazy'; image.width = 2466; image.height = 990;
    image.alt = '两组固定10秒音频在六种受控处理后的分数变化：左图AudioBox PQ、右图官方PAM，横轴独立；40Hz调制相对等RMS纯增益对照，其余相对原片段。零线表示无变化。';
    figure.append(image, el('figcaption', '两个内容、六种已知操作的自动响应；左侧PQ与右侧PAM各用独立横轴。40 Hz调制只与等RMS纯增益比较，其余与原片段比较。处理不等于人工缺陷标签；这些点不估计总体置信区间。'), link('查看SVG矢量图', './visuals/naturalness-response.svg'));
    root.append(details('受控反例与失败观察', [figure, el('p', '变换是已知操作；“是否更自然”仍需听感确认。分数反向或内容偏好不能单独证明reward hacking。', 'nat-note'), table(['对照', '实测变化', '解释边界'], comparisons.map((c) => [asText(first(c, ['label', 'name', 'comparison', 'id'])), asText(first(c, ['observation', 'result', 'finding', 'delta_summary', 'description'])), asText(first(c, ['interpretation', 'caveat', 'limitation', 'risk', 'note']))]))]));
  }
  const resources = Array.isArray(data.resources) ? data.resources : [];
  if (resources.length) root.append(details('速度与资源 · 本次实际运行', [table(['方法', '设备／运行范围', '耗时', '依赖／失败'], resources.map((r) => [asText(first(r, ['method', 'name', 'id'])), asText(first(r, ['device', 'hardware', 'scope'])), asText(first(r, ['timing', 'latency', 'runtime', 'elapsed_s', 'total_seconds', 'seconds'])), asText(first(r, ['notes', 'note', 'dependencies', 'failure', 'status']))]))]));
  const recommendations = Array.isArray(data.recommendations) ? data.recommendations : [];
  const recContent = recommendations.map((r) => { const article = el('article', null, 'nat-rec'); if (typeof r === 'string') article.append(el('p', r)); else { article.append(el('strong', asText(first(r, ['name', 'title', 'id'])))); for (const k of ['description', 'recommendation', 'formula', 'normalization', 'aggregation', 'weight', 'calibration', 'integration', 'status', 'risk', 'next_step']) if (r[k]) article.append(el('p', asText(r[k]), k === 'formula' ? 'nat-formula' : '')); } return article; });
  if (recContent.length) root.append(details('最值得继续的方案与RL接口', recContent));
  const references = Array.isArray(data.references) ? data.references : [];
  const referenceList = el('ol', null, 'nat-reference-list');
  for (const r of references) { const li = el('li'); if (typeof r === 'string') li.textContent = r; else { li.append(link(asText(first(r, ['title', 'name', 'id'])), first(r, ['url', 'paper_url', 'paper']))); for (const [key, title] of [['code', '官方代码'], ['code_url', '官方代码'], ['weights', '权重'], ['weights_url', '权重']]) if (r[key]) li.append(document.createTextNode(' · '), link(title, r[key])); if (r.note) li.append(el('span', ' — ' + asText(r.note))); } referenceList.append(li); }
  const reproduce = [];
  if (meta.report_url) reproduce.push(link('完整报告与候选核实', meta.report_url));
  if (meta.code_url) reproduce.push(document.createTextNode(' · '), link('评分代码与接口', meta.code_url));
  const commands = first(meta, ['reproduce_commands', 'commands']); if (commands) reproduce.push(el('pre', Array.isArray(commands) ? commands.join('\n') : asText(commands)));
  reproduce.push(el('p', `数据版本：${version}。试听为独立MP3副本；自动评分基于清单绑定的原始片段，哈希各自记录。人工标签只在你的浏览器本机保存，需导出后才能用于后续一致性分析。`, 'nat-note'));
  root.append(details('原始论文、官方代码与复现入口', [referenceList, ...reproduce]));
  updateOptions(); updatePlayer('A'); updatePlayer('B'); updatePair(); drawRanking();
}

if (root) load();
