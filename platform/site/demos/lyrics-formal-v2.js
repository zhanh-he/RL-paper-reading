import { createMilestones } from './chart-milestones.js';

const root = document.querySelector('#lyrics-formal');
const $ = (id) => root.querySelector(`#formal-v2-${id}`);
const keys = ['Coherence', 'Musicality', 'Memorability', 'Clarity', 'Naturalness'];
const colors = ['#219b76', '#407dc5', '#ce7240', '#b05496', '#7e8e30'];
const fmt = (value) => Number.isFinite(value) ? value.toFixed(4) : '未评测';
const delta = (value, baseline) => Number.isFinite(value) && Number.isFinite(baseline)
  ? `${fmt(value)} (${value >= baseline ? '+' : ''}${(value - baseline).toFixed(4)})` : fmt(value);
let registry, arm, index = 0, step = 270;
const rewardMarks = createMilestones($('reward'), 270);
const klMarks = createMilestones($('kl'), 270);

function button(label, action, selected, disabled = false) {
  const element = document.createElement('button');
  element.type = 'button'; element.textContent = label;
  element.setAttribute('aria-pressed', String(selected)); element.disabled = disabled;
  element.addEventListener('click', action);
  return element;
}

function chart(canvas, points, color, marks) {
  const width = Math.max(200, canvas.parentElement.clientWidth);
  const height = canvas.classList.contains('dimension-chart') ? 112 : 154;
  canvas.width = Math.round(width * devicePixelRatio); canvas.height = Math.round(height * devicePixelRatio);
  canvas.style.width = '100%'; canvas.style.height = `${height}px`;
  const context = canvas.getContext('2d'); context.scale(devicePixelRatio, devicePixelRatio);
  const rows = points.filter((p) => Number.isFinite(p.value));
  context.font = '11px system-ui'; context.fillStyle = '#666';
  if (!rows.length) { context.fillText('暂无已核验数据', 39, 65); return; }
  let low = Math.min(...rows.map((p) => p.value)); let high = Math.max(...rows.map((p) => p.value));
  const padding = Math.max((high - low) * 0.12, Math.abs(high) * 0.015, 0.00001);
  low = low >= 0 ? Math.max(0, low - padding) : low - padding; high += padding;
  const x = (s) => 39 + (width - 51) * s / 270;
  const y = (v) => 12 + (height - 41) * (high - v) / (high - low);
  for (let i = 0; i < 4; i++) {
    const value = low + (high - low) * i / 3;
    context.strokeStyle = '#e7e7e7'; context.beginPath(); context.moveTo(39, y(value)); context.lineTo(width - 12, y(value)); context.stroke();
    context.fillStyle = '#666'; context.fillText(value !== 0 && Math.abs(value) < 0.1 ? value.toExponential(1) : value.toFixed(2), 0, y(value) + 3);
  }
  for (const s of [0, 50, 100, 200, 270]) context.fillText(String(s), x(s) - 6, height - 6);
  function line(values, alpha, weight) {
    context.globalAlpha = alpha; context.strokeStyle = color; context.lineWidth = weight;
    context.beginPath(); values.forEach((p, i) => i ? context.lineTo(x(p.step), y(p.value)) : context.moveTo(x(p.step), y(p.value))); context.stroke(); context.globalAlpha = 1;
  }
  line(rows, 0.25, 1);
  line(rows.length < 20 ? rows : rows.map((p, i) => ({ step: p.step, value: rows.slice(Math.max(0, i - 4), i + 1).reduce((s, r) => s + r.value, 0) / Math.min(i + 1, 5) })), 1, 2);
  for (const p of rows.filter((p) => rows.length < 20 || [1, 5, 25, 50, 100, 270].includes(p.step))) {
    context.fillStyle = color; context.beginPath(); context.arc(x(p.step), y(p.value), 2.5, 0, Math.PI * 2); context.fill();
  }
  marks?.draw(context, width, height - 29, color);
}

function draw() {
  if (!arm) return;
  const curve = arm.train_curve || [];
  const selected = curve.filter((p) => [1, 5, 25, 50, 100, 270, ...(arm.checkpoints || [])].includes(p.step));
  rewardMarks.set(selected.map((p) => ({ step: p.step, label: `Step ${p.step} · training mean ${fmt(p.reward)}` })));
  klMarks.set(selected.map((p) => ({ step: p.step, label: `Step ${p.step} · sampled training KL ${fmt(p.sampled_kl)}` })));
  chart($('reward'), curve.map((p) => ({ step: p.step, value: p.reward })), colors[0], rewardMarks);
  chart($('kl'), curve.map((p) => ({ step: p.step, value: p.sampled_kl })), colors[1], klMarks);
  for (const canvas of $('dimensions').querySelectorAll('canvas')) {
    chart(canvas, Object.entries(arm.validation).map(([s, r]) => ({ step: Number(s), value: r.dimensions?.[canvas.dataset.key] })), colors[keys.indexOf(canvas.dataset.key)]);
  }
}

function row(values, heading = false) {
  const tr = document.createElement('tr');
  for (const value of values) { const td = document.createElement(heading ? 'th' : 'td'); td.textContent = value; tr.append(td); }
  return tr;
}

function tables() {
  const siblings = registry.arms.filter((a) => a.model === arm.model && a.reward === arm.reward);
  const stages = [...new Set([0, 270, ...siblings.flatMap((a) => Object.keys(a.validation).map(Number))])].sort((a, b) => a - b);
  $('validation-title').textContent = `${arm.model} · ${arm.reward} · 30 条固定验证，括号为 Δ vs 同臂 baseline`;
  $('validation-head').replaceChildren(row(['Step', ...siblings.map((a) => `LR ${a.lr}`)], true));
  $('validation').replaceChildren(...stages.map((s) => row([s, ...siblings.map((a) => delta(a.validation[s]?.mean_reward, a.validation[0]?.mean_reward))])));
  $('long-table').replaceChildren(...Object.entries(arm.long_replays || {}).map(([s, clips]) => {
    const scored = clips.filter((c) => Number.isFinite(c.reward?.mean));
    return row([s, scored.length === 3 ? fmt(scored.reduce((total, c) => total + c.reward.mean, 0) / 3) : '未评分',
      clips.length ? `${Math.min(...clips.map((c) => c.seconds)).toFixed(1)}–${Math.max(...clips.map((c) => c.seconds)).toFixed(1)} s` : '—',
      clips.length ? Math.max(...clips.map((c) => c.signal.peak)).toFixed(3) : '—',
      clips.length ? `${(100 * clips.reduce((total, c) => total + c.signal.near_full_scale_fraction, 0) / clips.length).toFixed(3)}%` : '—', `${scored.length}/3`]);
  }));
}

function pauseOthers(audio) {
  document.querySelectorAll('audio').forEach((other) => { if (other !== audio) other.pause(); });
}

function panel(clip, label, role) {
  const article = document.createElement('article'); article.className = 'compare-panel'; article.dataset.role = role;
  const title = document.createElement('h3'); title.textContent = label; article.append(title);
  if (!clip) {
    const p = document.createElement('p'); p.textContent = arm.model === 'Muse' && arm.standalone_inference === false
      ? '按当前实验安排暂停 Muse 独立推理；此处不放替代音频。'
      : '尚未回收该 checkpoint 的长片段；没有替代音频。'; article.append(p); return article;
  }
  article.dataset.sha256 = clip.sha256;
  const caption = document.createElement('p'); caption.className = 'muted';
  caption.textContent = `${clip.seconds.toFixed(2)} s · seed ${clip.seed} · MP3 192 kbps${clip.seconds < 30 ? ' · 未达到 30 秒目标' : ''}${clip.truncated ? ' · 达到生成上限' : ''}`; article.append(caption);
  const audio = document.createElement('audio'); audio.controls = true; audio.preload = 'metadata'; audio.src = clip.audio;
  audio.addEventListener('play', () => pauseOthers(audio));
  audio.addEventListener('error', () => { $('listen-status').textContent = '音频加载失败，请使用 MP3 链接重试。'; }); article.append(audio);
  for (const [type, alt] of [['wave', 'Waveform · fixed −1 to +1'], ['spectrum', 'Mel · 0–24 kHz · fixed −80 to 0 dB']]) {
    const figure = document.createElement('figure'); figure.className = type === 'wave' ? 'wave-visual' : 'spectrum-visual';
    const img = document.createElement('img'); img.src = clip[type]; img.alt = alt; img.loading = 'lazy'; figure.append(img); article.append(figure);
  }
  const link = document.createElement('a'); link.href = clip.audio; link.textContent = '下载 MP3'; article.append(link);
  return article;
}

function replay() {
  const baseline = arm.long_replays?.['0']?.find((c) => c.index === index);
  const candidate = arm.long_replays?.[String(step)]?.find((c) => c.index === index);
  const grid = $('compare'); const oldA = grid.querySelector('[data-role="baseline"]');
  if (!oldA || oldA.dataset.sha256 !== baseline?.sha256) {
    oldA?.querySelector('audio')?.pause(); const next = panel(baseline, `A · Baseline (${arm.model})`, 'baseline');
    if (oldA) oldA.replaceWith(next); else grid.append(next);
  }
  grid.querySelector('[data-role="candidate"]')?.querySelector('audio')?.pause();
  const nextB = panel(candidate, `B · ${arm.model}-${arm.reward}-${arm.lr} · Step ${step}`, 'candidate');
  const oldB = grid.querySelector('[data-role="candidate"]'); if (oldB) oldB.replaceWith(nextB); else grid.append(nextB);
  const stages = [...new Set([0, ...(arm.checkpoints || []), ...Object.keys(arm.long_replays || {}).map(Number)])].sort((a, b) => a - b);
  $('stages').replaceChildren(...stages.map((s) => button(`${s} step${s === 1 ? '' : 's'}`, () => { step = s; replay(); }, step === s,
    !arm.long_replays?.[s]?.some((c) => c.index === index))));
  $('replay-note').textContent = arm.model === 'Muse' && arm.standalone_inference === false
    ? 'Muse 本轮只提交正式训练，保留 GRPO 内部采样；独立推理和验证生成暂停。'
    : '相同条件、seed 和冻结起点；长片段为独立推理，不代替 30 条短片段验证。YuE2 上限 1600 semantic tokens，最少 200，实际时长逐文件报告。波形 ±1，Mel 0–24 kHz / −80–0 dB，黑色底。';
  root.querySelectorAll('[data-formal-listen]').forEach((b) => { b.disabled = !(b.dataset.formalListen === 'baseline' ? baseline : candidate); });
  $('listen-status').textContent = '';
}

function select(next) {
  arm = next;
  $('controls').querySelectorAll('button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.id === arm.id)));
  $('status').textContent = arm.status === 'verified-complete' ? '270 步 + 固定验证已核验'
    : arm.status === 'queued' ? '正式训练已提交 · 排队中'
    : `${arm.verified_train_steps} 步已核验${arm.status.includes('failed') ? ' · 中途失败' : ' · 固定验证尚待回收'}`;
  $('status').className = `state ${arm.status === 'verified-complete' ? 'measured' : 'diagnostic'}`;
  $('description').textContent = `${arm.model}-${arm.reward}-${arm.lr} · ${arm.host}${arm.job_id ? ` · ${arm.project} · ${arm.job_id} · priority ${arm.priority}` : ''} · ${arm.failure_reason || '训练 / 验证与新客观 test 表分开；结果不代表已听辨的音质提升。'}`;
  $('reward-title').textContent = `优化目标 · ${arm.reward} 五维均值`;
  $('reward-definition').textContent = arm.reward === 'SongEval'
    ? 'R = (Coherence + Musicality + Memorability + Clarity + Naturalness) / 5。SongEval 使用 MuQ 音频编码器与评分头评估音乐审美；coverage、乐器 richness、beat、歌词匹配、时长与削波没有独立 reward。训练与短片段验证使用 WAV 原件评分；网页只提供 MP3 试听副本。短片段分数不归到独立长音频上。'
    : 'R = (Coherence + Musicality + Memorability + Clarity + Naturalness) / 5。MuseCritic 的含义为整体连贯性、整体音乐性、可记忆性、歌曲结构清晰度、人声呼吸与分句自然度；先生成评论，再用评分头预测。它不是 SongEval，也不与其数值横比。本实验评分私有归档的 PCM24-FLAC 原件；公开音频只使用 MP3 试听副本，不宣称压缩文件就是评分原件。';
  $('train-protocol').textContent = `CMI-Pref：270 train / 30 valid / 121 official test；test 2026-10-13 开始独立评估，不参与调参。投影 text + lyrics，不输入参考音频。${arm.model === 'YuE2' ? 'PyTorch + PEFT LoRA，组大小 2，AdamW；600 semantic tokens。' : 'ms-swift / TRL + vLLM，LoRA，组大小 2；512 completion tokens。从冻结基座重启；无 AUDIO、非有限值或不足 1 秒的解码输出记 0 分并单独记录，基础设施错误停机。'} β = 0.01，记录对冻结参考策略的 sampled training KL。`;
  $('reward-caption').textContent = 'Reward · 五维均值 · 逐步训练值 + trailing 5-step mean';
  $('kl').previousElementSibling.querySelector('span').textContent = 'KL · 训练期间冻结参考策略采样估计';
  $('reward-value').textContent = fmt(arm.train_curve.at(-1)?.reward);
  $('kl-value').textContent = fmt(arm.train_curve.at(-1)?.sampled_kl);
  $('dimensions-note').textContent = '30 条固定验证的各维均值；仅展示实际评分，不从 mean 推算子分数。';
  $('dimensions').replaceChildren(...keys.map((key) => {
    const figure = document.createElement('figure'); const cap = document.createElement('figcaption'); cap.textContent = key;
    const canvas = document.createElement('canvas'); canvas.className = 'dimension-chart'; canvas.dataset.key = key; figure.append(cap, canvas); return figure;
  }));
  $('framework').textContent = 'YuE2：原生 PyTorch + PEFT，每组只更新一次，该次梯度中的 ratio clipping 不起作用；log-prob 使用原始模型分布，生成使用原生采样变换，右图是已记录的采样 KL 正则项，不是精确全策略 KL。Muse：ms-swift / TRL + vLLM。长音频采用同一模型解码，不循环、拼接独立歌曲或补静音。';
  $('replicates').replaceChildren(...(registry.replicates || []).map((a) => row([`${a.model}-${a.reward}-${a.lr}`, a.host, a.verified_train_steps, `${fmt(a.validation[0]?.mean_reward)} → ${fmt(a.validation[270]?.mean_reward)}`])));
  $('superseded').replaceChildren(...(registry.superseded_arms || []).map((a) => row([
    `${a.model}-${a.reward}-${a.lr}`, a.verified_train_steps,
    a.status.includes('failed') ? '部分失败' : '训练完成，验证未完成', a.failure_reason || '已被同 LR 重跑替代'
  ])));
  const available = Object.keys(arm.long_replays || {}).map(Number).filter((s) => s > 0);
  step = available.includes(step) ? step : Math.max(0, ...available);
  tables(); replay(); draw();
}

try {
  const response = await fetch('./formal-9to1-runs.json', { cache: 'no-store' });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  registry = await response.json();
  if (registry.schema_version !== 3 || registry.train_conditions !== 270 || registry.valid_conditions !== 30 || registry.sealed_test_conditions !== 121) throw new Error('Formal protocol mismatch');
  $('controls').replaceChildren(...registry.arms.map((a) => {
    const b = button(`${a.model}-${a.reward}-${a.lr}${a.status.includes('failed') ? ' · 部分失败' : ''}`, () => select(a), false); b.dataset.id = a.id; return b;
  }));
  const showExample = (example) => {
    index = example.index; $('style').textContent = example.style; $('lyrics').textContent = example.lyrics;
    $('seed').textContent = `valid ${index} · seed ${example.seed}`;
    $('examples').querySelectorAll('button').forEach((b) => b.setAttribute('aria-pressed', String(Number(b.dataset.index) === index)));
    if (arm) replay();
  };
  $('examples').replaceChildren(...registry.examples.map((example, i) => {
    const b = button(`Example ${i + 1}`, () => showExample(example), i === 0); b.dataset.index = example.index; return b;
  }));
  showExample(registry.examples[0]); select(registry.arms[0]);
  root.querySelectorAll('[data-formal-listen]').forEach((b) => b.addEventListener('click', async () => {
    const audio = $('compare').querySelector(`[data-role="${b.dataset.formalListen}"] audio`);
    if (!audio) { $('listen-status').textContent = '该片段尚未回收。'; return; }
    const other = [...$('compare').querySelectorAll('audio')].find((a) => a !== audio);
    const position = other?.currentTime || 0; pauseOthers(audio);
    try { if (audio.readyState < 1) await new Promise((resolve, reject) => { audio.addEventListener('loadedmetadata', resolve, { once: true }); audio.addEventListener('error', reject, { once: true }); });
      audio.currentTime = Math.min(position, Math.max(0, audio.duration - 0.1)); await audio.play(); $('listen-status').textContent = `正在播放 ${b.dataset.formalListen === 'baseline' ? 'A' : 'B'}`;
    } catch { $('listen-status').textContent = '浏览器未能播放，请检查 MP3 链接。'; }
  }));
  new ResizeObserver(draw).observe($('reward').parentElement);
  $('dimensions-details').addEventListener('toggle', draw);
} catch (error) { $('status').textContent = '实验清单加载失败'; console.error(error); }
