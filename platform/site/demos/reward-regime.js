import { createMilestones } from './chart-milestones.js';

const root = document.querySelector('#view-reward-regimes');
const $ = (id) => root.querySelector(`[data-regime="${id}"]`);
const checkpoints = [0, 50, 100, 270];
const rewardKey = (id) => ({ songbench: 'instrument', 'songbench-arrangement': 'arrangement' })[id] || id;
const rewardName = (key) => ({ richness: 'SP-Richness', instrument: 'SongBench-Instrument', arrangement: 'SongBench-Arrangement' })[key] || key;
const statusNames = { running: '运行中', 'local-sequential-queue': '连续队列', 'pbs-queued': 'PBS 排队 · 未起跑', 'preparation-submitted': '准备链已提交 · 非GPU训练', complete: '已核验完成', 'head-failed': 'Head-only 失败 · 待替代', 'head-stopped': 'Head-only 已停止 · 待替代', 'head-cancelled': 'Head-only 未启动 · 已取消' };
const marks = new Map(['reward', 'kl'].map((key) => [key, createMilestones($(`${key}-chart`), 270)]));
const fmt = (value) => Number.isFinite(value) ? value.toFixed(4) : 'n/a';
const fmtKL = (value) => Number.isFinite(value) && value !== 0 && Math.abs(value) < .0001 ? value.toExponential(2) : fmt(value);
const delta = (value, baseline) => Number.isFinite(value) && Number.isFinite(baseline)
  ? `${fmt(value)} (${value >= baseline ? '+' : ''}${fmt(value - baseline)})` : fmt(value);
let protocol, progress, recovery, examples;
let selected = 'yue2', target = 'richness', replayMode = 'long', example = 0, replayStep = 270, badId = 0;

const model = () => protocol.models.find((m) => m.id === selected);
const experiment = () => `${model().name}-${rewardName(target)}-3e-4`;
const measuredRun = (key = target) => progress.runs.find((r) => r.model === selected && r.reward === key);
const selectedRecovery = () => recovery.model === selected ? recovery : recovery.additional_models?.[selected];

function row(values, heading = false) {
  const tr = document.createElement('tr');
  for (const value of values) {
    const cell = document.createElement(heading ? 'th' : 'td');
    if (value instanceof Node) cell.append(value); else cell.textContent = value;
    tr.append(cell);
  }
  return tr;
}

function button(value, label, current, choose, disabled = false) {
  const element = document.createElement('button');
  element.type = 'button'; element.textContent = label; element.disabled = disabled;
  if (label.endsWith('-3e-4')) {
    element.textContent = label.slice(0, -4);
    const lr = document.createElement('span'); lr.className = 'regime-lr'; lr.textContent = '3e-4'; element.append(lr);
  }
  element.setAttribute('aria-pressed', String(value === current));
  element.addEventListener('click', () => choose(value));
  return element;
}

function controls(container, items, current, choose) {
  container.replaceChildren(...items.map(([value, label, disabled]) => button(value, label, current, choose, disabled)));
}

function chart(key, points) {
  const canvas = $(`${key}-chart`), width = canvas.clientWidth, height = canvas.clientHeight;
  if (!width || !height) return;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr);
  const ctx = canvas.getContext('2d'); ctx.scale(dpr, dpr);
  const values = points.map((p) => p[key]);
  let low = Math.min(0, ...values), high = Math.max(...values);
  if (high - low < .00001) high = low + .00001;
  const padding = .12 * (high - low); high += padding; if (low < 0) low -= padding;
  const axisY = height - 30, color = key === 'reward' ? '#177e70' : '#407dc5';
  const x = (step) => 39 + (width - 51) * step / 270;
  const y = (value) => 12 + (axisY - 12) * (high - value) / (high - low);
  ctx.font = '11px system-ui'; ctx.lineWidth = 1;
  for (let i = 0; i < 4; i++) {
    const value = low + (high - low) * i / 3, yy = y(value);
    ctx.strokeStyle = '#dde4df'; ctx.beginPath(); ctx.moveTo(39, yy); ctx.lineTo(width - 12, yy); ctx.stroke();
    ctx.fillStyle = '#59615c'; ctx.textAlign = 'right';
    ctx.font = key === 'kl' ? '10px system-ui' : '11px system-ui';
    const axisLabel = key === 'kl' ? value === 0 ? '0' : Math.abs(value) < .001 ? value.toExponential(1) : value.toFixed(3) : value.toFixed(2);
    ctx.fillText(axisLabel, 35, yy + 4);
  }
  ctx.textAlign = 'center'; ctx.fillStyle = '#59615c';
  for (const step of [0, 50, 100, 200, 270]) ctx.fillText(String(step), x(step), height - 8);
  function line(rows, opacity, weight) {
    ctx.globalAlpha = opacity; ctx.strokeStyle = color; ctx.lineWidth = weight; ctx.beginPath();
    rows.forEach((p, i) => i ? ctx.lineTo(x(p.step), y(p.value)) : ctx.moveTo(x(p.step), y(p.value)));
    ctx.stroke(); ctx.globalAlpha = 1;
  }
  line(points.map((p) => ({ step: p.step, value: p[key] })), .22, 1);
  line(points.map((p, i) => ({ step: p.step, value: points.slice(Math.max(0, i - 9), i + 1)
    .reduce((sum, q) => sum + q[key], 0) / Math.min(i + 1, 10) })), 1, 2);
  if (replayStep <= points.at(-1).step) {
    ctx.strokeStyle = '#79827d'; ctx.lineWidth = 1; ctx.setLineDash([3, 4]);
    ctx.beginPath(); ctx.moveTo(x(replayStep), 12); ctx.lineTo(x(replayStep), axisY); ctx.stroke(); ctx.setLineDash([]);
  }
  marks.get(key).draw(ctx, width, axisY, color);
}

function draw() {
  if (!progress) return;
  const run = measuredRun(), trace = run?.trace || [];
  $('plots').hidden = !trace.length;
  for (const key of ['reward', 'kl']) {
    marks.get(key).set(trace.filter((p) => [1, 5, 25, 50, 100, 270].includes(p.step)).map((p) => ({
      step: p.step, label: `Step ${p.step}\n${key === 'kl' ? 'KL' : 'Reward'} ${key === 'kl' ? fmtKL(p[key]) : p[key].toFixed(6)}（日志原值）`,
    })));
    $(`${key}-value`).textContent = trace.length ? `末步 ${key === 'kl' ? fmtKL(trace.at(-1)[key]) : fmt(trace.at(-1)[key])}` : '';
    if (trace.length) chart(key, trace);
  }
  $('trace-note').textContent = trace.length
    ? `${run.steps}/270 步已核验。${model().readiness === 'ablation' ? 'Head-only消融已停止，待主干LoRA替代。' : ''}浅线为原值，实线为10步滚动均值；Reward不含KL loss。${selected === 'diffrhythm2' ? '右图为采样状态上解析条件高斯KL，按latent维度归一化；不是波形KL。' : ['heartmula','songbloom'].includes(selected) ? '右图为head动作分布的解析categorical KL。' : '右图为训练rollout上的冻结参考采样估计。'} 这不是离线探针或测试分数。`
    : `${model().status}。尚无本模型、本目标的已核验日志，不显示替代曲线。`;
}

function pauseOthers(audio) {
  document.querySelectorAll('audio').forEach((other) => { if (other !== audio) other.pause(); });
}

function panel(asset, label, role) {
  const article = document.createElement('article'); article.className = 'compare-panel'; article.dataset.role = role;
  article.dataset.binding = `${selected}:${asset?.sha256 || 'pending'}`;
  const h = document.createElement('h3'); h.textContent = label; article.append(h);
  if (!asset) {
    const note = document.createElement('p'); note.className = 'regime-placeholder';
    note.textContent = selected === 'private' ? '私有对照未公开音频、权重或实验结果。'
      : role === 'baseline' ? '尚未回收本模型的冻结起点音频。' : '尚未回收该实验与步骤的GRPO音频，不借用其他模型或实验。';
    article.append(note); return article;
  }
  const caption = document.createElement('p'); caption.className = 'muted';
  caption.textContent = `${asset.seconds.toFixed(2)} s · MP3 192 kbps / 48 kHz · 无响度归一化`; article.append(caption);
  const audio = document.createElement('audio'); audio.controls = true; audio.preload = 'metadata'; audio.src = asset.audio;
  audio.dataset.sourceSha = asset.source_audio_sha256;
  audio.addEventListener('play', () => {
    pauseOthers(audio);
    root.querySelectorAll('[data-regime-listen]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.regimeListen === role)));
    $('listen-status').textContent = `正在播放 ${role === 'baseline' ? 'A · Baseline' : 'B · GRPO'}`;
  });
  audio.addEventListener('pause', () => {
    root.querySelector(`[data-regime-listen="${role}"]`).setAttribute('aria-pressed', 'false');
  });
  audio.addEventListener('error', () => { $('listen-status').textContent = '音频加载失败，请核对MP3链接。'; });
  article.append(audio);
  for (const [type, alt] of [['wave', 'Waveform · fixed -1 to +1'], ['spectrum', 'Mel · 0-24 kHz · fixed -80 to 0 dB · black floor']]) {
    const figure = document.createElement('figure'); figure.className = type === 'wave' ? 'wave-visual' : 'spectrum-visual';
    const img = document.createElement('img'); img.src = asset[type]; img.alt = alt; img.loading = 'lazy'; figure.append(img); article.append(figure);
  }
  const link = document.createElement('a'); link.href = asset.audio; link.textContent = '下载 MP3'; link.className = 'audio-fallback';
  const scale = document.createElement('small'); scale.textContent = 'Wave ±1 · Mel 0–24kHz / −80–0dB';
  article.append(link, scale); return article;
}

function compare(a, b) {
  // Preserve the exact baseline element across reward/checkpoint changes.
  for (const [role, asset, label] of [['baseline', a, `A · Baseline (${model().name})`],
    ['candidate', b, `B · ${experiment()} · Step ${replayStep}`]]) {
    const previous = $('media').querySelector(`[data-role="${role}"]`);
    if (role === 'baseline' && previous?.dataset.binding === `${selected}:${asset?.sha256 || 'pending'}`) continue;
    previous?.querySelector('audio')?.pause();
    const next = panel(asset, label, role);
    if (previous) previous.replaceWith(next); else $('media').append(next);
  }
  for (const element of root.querySelectorAll('[data-regime-listen]')) {
    element.disabled = !(element.dataset.regimeListen === 'baseline' ? a : b);
    const audio = $('media').querySelector(`[data-role="${element.dataset.regimeListen}"] audio`);
    element.setAttribute('aria-pressed', String(Boolean(audio && !audio.paused)));
  }
  if (![...$('media').querySelectorAll('audio')].some((audio) => !audio.paused)) $('listen-status').textContent = '';
  $('pair-metrics').replaceChildren(); $('audio-provenance').replaceChildren();
  if (!a || !b) {
    $('pair-metrics').append(row(['尚无完整A/B评分对', 'n/a', 'n/a'])); return;
  }
  const value = (metrics, key) => key === 'objective'
    ? metrics.invalid ? 0 : (target === 'richness' ? metrics.SP_Richness : metrics[target === 'instrument' ? 'Instrumental' : 'Arrangement'] / 10) - (metrics.penalty ?? 0)
    : metrics[key];
  $('pair-metrics').replaceChildren(...[['objective', 'Target − shared penalty'], ['SP_Richness', 'SP Richness · 0–1'],
    ['Instrumental', 'Instrumental · 0–10'], ['Arrangement', 'Arrangement · 0–10'], ['Musicality', 'Musicality · 0–10'],
    ['Mixing', 'Mixing · 0–10'], ['clipping_fraction', '近满幅样本比例']].map(([key, label]) =>
    row([label, fmt(value(a.metrics, key)), delta(value(b.metrics, key), value(a.metrics, key))])));
  for (const [label, asset] of [['A', a], ['B', b]]) {
    const note = document.createElement('p');
    note.textContent = `${label} · 原件 SHA256 ${asset.source_audio_sha256} · MP3 SHA256 ${asset.sha256}`;
    $('audio-provenance').append(note);
  }
}

function renderMedia() {
  if (!recovery) return;
  root.querySelectorAll('[data-mode]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.mode === replayMode)));
  $('seed').textContent = '';
  $('replay-title').textContent = replayMode === 'long' ? selected === 'diffrhythm2' ? 'Head-only 消融回放 · 原生60秒预算' : 'Long-form replay · target 60 seconds' : 'Badcase · 原始短验证片段';
  $('input').textContent = model().input;
  $('reference-note').textContent = selected === 'private' ? '未公开输入或生成结果。'
    : selected === 'songbloom' ? '原生使用固定参考音频与歌词；准备链须校验全部300条参考资源后才启动训练。不公开原始reference；它不是唯一正确输出。'
      : '本轮不输入参考音频；各reward共享同模型的冻结起点。';
  const recovered = selectedRecovery();
  let a, b, inputRow;
  if (replayMode === 'long') {
    const available = recovered?.long_replays.filter((r) => r.reward === target) || [];
    const baselines = recovered?.baseline || [];
    const indices = examples.display_indices_by_model?.[selected] || examples.display_indices;
    if (!indices.includes(example)) example = indices[0];
    controls($('example-controls'), indices.map((i, n) => {
      const count = examples.inputs.find((r) => r.index === i).requested_count;
      return [i, `Sample ${n + 1}${count ? ` · ${count} 类` : ''}`];
    }), example, (i) => { example = i; renderMedia(); });
    inputRow = examples.inputs.find((r) => r.index === example);
    a = baselines.find((r) => r.index === example)?.asset;
    if (model().readiness === 'ablation' && replayStep !== 0 && available.length && !available.some((r) => r.step === replayStep)) {
      replayStep = Math.max(...available.map((r) => r.step));
    }
    const steps = [...new Set([...checkpoints, ...available.map((r) => r.step)])].sort((x, y) => x - y);
    controls($('step-controls'), steps.map((s) => [s, `${s} steps`, s === 0 ? !a : !available.some((r) => r.index === example && r.step === s)]),
      replayStep, (s) => { replayStep = s; renderMedia(); });
    b = replayStep === 0 ? a : available.find((r) => r.index === example && r.step === replayStep)?.asset;
    $('input-note').textContent = inputRow.kind === 'instrument-count-probe'
      ? `改写试听探针 · 原始 valid #${inputRow.base_valid_index + 1} · seed ${inputRow.seed} · 不计入全30条验证`
      : `CMI valid #${example + 1} · 原始条件 · seed ${inputRow.seed}`;
    $('seed').textContent = `seed ${inputRow.seed}${a ? '' : ' · 音频待回收'}`;
    $('replay-note').textContent = b
      ? selected === 'diffrhythm2' ? '旧head-only消融，待主干LoRA替代；不是270步完成的主干实验。这里回收的就是0/50/100步完整验证中的已评分原始音频，不额外生成或补齐。冻结baseline与当前checkpoint条件和seed一致，均使用同一SDE基线。图像与试听来自MP3，分数绑定lossless原件。'
      : '冻结baseline与当前checkpoint使用同一条件和seed。原生长回放，不循环或补齐；训练/全量短验证的分数不归到这段长音频。图像来自MP3，评分绑定原件。'
      : `${model().status}。${a ? 'A保留已核验的同模型baseline；B长回放尚未回收。' : '音频和可视化均未用其他实验替代。'}`;
  } else {
    const cases = recovered?.badcases.filter((r) => r.reward === target) || [];
    if (badId >= cases.length) badId = 0;
    controls($('example-controls'), cases.map((r, i) => [i, `Valid #${r.index + 1} · step ${r.step}`]), badId, (i) => { badId = i; renderMedia(); });
    const item = cases[badId];
    if (item) {
      inputRow = examples.inputs.find((r) => r.index === item.index);
      a = item.baseline; b = item.candidate; replayStep = item.step;
      controls($('step-controls'), [[item.step, `${item.step} steps`]], replayStep, () => {});
      $('seed').textContent = `CMI valid #${item.index + 1} · seed ${item.seed}`;
      $('input-note').textContent = `固定验证 #${item.index + 1} · 事后选取的proxy-conflict候选`;
      $('replay-note').textContent = `${item.step < 270 ? `这是step${item.step}的归档中间结果，不是270步终点。` : ''}确实被评分的短验证音频，不是30–60秒回放。刻意选择目标上涨、交叉分数下降的样本；尚未由盲听确认reward hacking。选择规则与全30条分布见下方折叠区。`;
    } else {
      $('step-controls').replaceChildren(); $('input-note').textContent = '本实验尚无已归档的badcase音频对。';
      $('replay-note').textContent = '没有已回收的指标冲突候选；不把“尚未回收”解释为没有reward hacking。';
    }
  }
  $('text-content').textContent = selected === 'private' ? '私有对照输入未公开。' : inputRow?.style || '本实验暂无已归档样本。';
  $('lyrics-content').textContent = selected === 'private' ? '未公开。' : inputRow?.lyrics || 'n/a';
  $('count-note').textContent = selected === 'songbloom'
    ? 'Text 为数据集原始条件，但此版本不输入Text；乐器数量提示遵循为 n/a。这里固定五个独立原生条件（valid0/10/20/1/2），不拿相同音频重复充数。'
    : inputRow?.requested_count
    ? `恰好 ${inputRow.requested_count} 类伴奏乐器（不含人声；鼓组算一类）。Sample 4/5 使用相同原始歌词和 seed，仅改写配器要求；这是提示遵循探针，不代表已测得真实乐器数量。`
    : 'Sample 1–3 保留原始提示，不额外规定乐器数量；其中部分提示列出了乐器。Sample 4/5 明确规定数量，单独作为试听探针。';
  const attribution = document.createElement('a'); attribution.href = examples.attribution.url;
  attribution.textContent = 'CMI-Pref · Haiwen Xia et al. · CC BY-NC-SA 4.0';
  $('input-attribution').replaceChildren(attribution);
  compare(a, b); draw();
}

function tables() {
  $('arms').replaceChildren(...protocol.rewards.map((reward) => {
    const key = rewardKey(reward.id), run = protocol.runs.find((r) => r.model === selected && r.reward === key), data = measuredRun(key);
    const baseline = data?.validation.find((v) => v.step === 0), latest = data?.validation.at(-1);
    const valid = latest && baseline ? `Step ${latest.step} · ${delta(latest.mean, baseline.mean)} · ${latest.n}/30` : '待全量30条验证';
    return row([`${model().name}-${rewardName(key)}-3e-4`, run
      ? `${data?.complete ? '已核验完成' : statusNames[run.status] || run.status} · ${run.host}${run.project ? ` / ${run.project}` : ''}${run.job_id ? ` · ${run.job_id}` : run.preparation_job_id ? ` · 准备任务 ${run.preparation_job_id}` : ''}`
      : model().status, run ? `${data?.steps || 0}/270` : '未提交', run ? valid : 'n/a', '封存 · 未运行']);
  }));
  const run = measuredRun(), baseline = run?.validation.find((v) => v.step === 0);
  $('validation-title').textContent = `${experiment()} · 固定验证 / 30 conditions`;
  $('validation').replaceChildren(...checkpoints.map((step) => {
    const data = run?.validation.find((v) => v.step === step);
    return row([`${step} · ${data ? `${data.n}/30` : '未评测'}`, delta(data?.mean, baseline?.mean),
      ...['SP_Richness', 'Instrumental', 'Arrangement', 'Mixing', 'Musicality'].map((key) => delta(data?.cross_means?.[key], baseline?.cross_means?.[key])),
      Number.isInteger(data?.invalid_n) ? `${data.invalid_n}/30` : 'n/a']);
  }));
  const interval = (value) => `${value.mean_delta >= 0 ? '+' : ''}${fmt(value.mean_delta)} [${value.ci95.map(fmt).join(', ')}]`;
  const audits = selectedRecovery()?.audits.filter((r) => r.reward === target) || [];
  $('case-audit').replaceChildren(...audits.map((r) => row([`${rewardName(r.reward)} · Step ${r.step}${r.complete_arm ? '' : ' · 中间checkpoint'}`,
    `${r.candidate_n}/${r.n}`, ...['objective', 'Instrumental', 'Arrangement'].map((key) => interval(r.statistics[key])), r.selection])));
  if (!audits.length) $('case-audit').append(row(['尚未回收本实验的候选筛选记录', 'n/a', 'n/a', 'n/a', 'n/a', '未用其他模型替代']));
}

function render() {
  for (const b of $('models').querySelectorAll('button')) b.setAttribute('aria-pressed', String(b.dataset.model === selected));
  controls($('reward-controls'), protocol.rewards.map((r) => [rewardKey(r.id), `${model().name}-${rewardName(rewardKey(r.id))}-3e-4`]),
    target, (value) => { target = value; render(); });
  const estimate = model().team_listening_estimate;
  $('model-note').textContent = `${model().readiness === 'ablation' ? 'Head-only 消融 · 待主干LoRA替代 · ' : ''}${model().role}${Number.isFinite(estimate) ? ` · 团队十输入快听 ≈${estimate.toFixed(1)}（非正式MOS）` : ''} · 本站正式MOS 未测`;
  const run = measuredRun(), submitted = protocol.runs.find((r) => r.model === selected && r.reward === target);
  $('status').textContent = run?.complete ? '270步 + 30/30验证已核验' : submitted ? statusNames[submitted.status] || submitted.status : model().status;
  $('status').className = `state ${run?.complete ? 'measured' : 'pending'}`;
  const reward = protocol.rewards.find((r) => rewardKey(r.id) === target);
  $('reward-title').textContent = `优化目标 · ${reward.name}`;
  $('reward-definition').textContent = `${reward.definition}。${target === 'richness'
    ? '伴奏频谱活动/变化与tonality gate，不是乐器数量或MOS。'
    : target === 'instrument' ? '评乐器真实度与合成质量，不直接评乐器数量。' : '评和声框架与配器编排，不是简单计数。'} 三臂都使用同一削波/RMS guard；KL为独立loss项。`;
  $('train-protocol').textContent = ['yue2', 'muse'].includes(selected)
    ? `CMI-Pref 270 train / 30 valid · 恒定LR3e-4 · seed42 · 组大小2 · ${selected === 'yue2' ? '历史轮 β0.01。PyTorch + PEFT，600 semantic-token短训练预算；长回放单独推理。' : `新轮 β0.05。原始非GRPO Muse，新主干LoRA；ms-swift / TRL + colocated vLLM，512 completion-token /10.24s解码。${submitted?.status==='running'?'5090正式训练中；完整验证未回收。':`Gadi正式任务排队：${submitted?.job_id||'未提交'}。`}`}`
    : submitted ? `CMI-Pref 270 train / 30 valid · LR3e-4 · G2 · β0.01 · 原生60秒预算，允许自然提前结束。${submitted.training_scope}；rank8/alpha16，仅训练输出头，非完整主干。KL单位：${submitted.kl_unit}。`
      : `${model().input} · ${model().status}。暂无已生效训练配置。`;
  $('kl-caption').textContent = selected === 'diffrhythm2' ? 'KL · 条件高斯 / latent dimension' : ['heartmula','songbloom'].includes(selected) ? 'KL · 输出头 categorical / action' : 'KL · 在线冻结参考采样估计';
  $('backbone-status').textContent = `新主干轮 · 恒定LR3e-4（与当前YuE2组一致）· β0.05 · ${protocol.backbone_round.status_by_model[selected]}`;
  $('backbone-report').href = protocol.backbone_round.report_url;
  $('model-detail').textContent = `${model().name}：${model().note}`;
  $('model-source').hidden = !model().source;
  if (model().source) $('model-source').href = model().source; else $('model-source').removeAttribute('href');
  tables(); renderMedia();
}

try {
  async function load(path) {
    const response = await fetch(path, { cache: 'no-store' });
    if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
    return response.json();
  }
  protocol = await load('./data/reward-regime-protocol.json');
  progress = await load('./data/reward-regime-progress.json');
  recovery = await load('./data/reward-regime-recovery.json');
  examples = await load('./data/reward-regime-examples.json');
  if (!protocol.training_authorized || protocol.runs.length < 6) throw new Error('Approved executed matrix mismatch');
  $('updated').textContent = `已核验快照 ${new Date(progress.updated_at).toLocaleString('zh-CN', { timeZone: 'Australia/Perth' })} AWST · 不是远端实时状态。准备链、排队和真实更新分开标注；未回收日志不显示替代曲线。`;
  $('hypothesis').textContent = '研究问题：不同起点模型对SP与AI reward的响应是否不同？团队快听只提供分层假设；自动分数与同事的十输入估分均不是本站正式MOS或可听提升结论。';
  $('models').replaceChildren(...protocol.models.map((m) => {
    const b = button(m.id, m.name, selected, (value) => { selected = value; replayStep = 270; render(); }); b.dataset.model = m.id; return b;
  }));
  $('rewards').replaceChildren(...protocol.rewards.map((r) => {
    const link = document.createElement('a'); link.href = r.source; link.textContent = r.name;
    return row([link, r.definition, r.meaning, r.note]);
  }));
  $('data').replaceChildren(...['train', 'valid', 'test', 'modalities'].map((key) => row([key, protocol.data[key]])));
  $('design').replaceChildren(...Object.entries(protocol.design).map(([key, value]) => row([key, Array.isArray(value) ? value.join(' / ') : value])));
  root.querySelectorAll('[data-mode]').forEach((b) => b.addEventListener('click', () => { replayMode = b.dataset.mode; replayStep = 270; renderMedia(); }));
  root.querySelectorAll('[data-regime-listen]').forEach((b) => b.addEventListener('click', async () => {
    const audio = $('media').querySelector(`[data-role="${b.dataset.regimeListen}"] audio`);
    if (!audio) return;
    const playing = [...$('media').querySelectorAll('audio')].find((a) => !a.paused);
    const position = playing?.currentTime || audio.currentTime || 0; pauseOthers(audio);
    try {
      if (audio.readyState < 1) await new Promise((resolve, reject) => {
        audio.addEventListener('loadedmetadata', resolve, { once: true }); audio.addEventListener('error', reject, { once: true }); audio.load();
      });
      audio.currentTime = Math.min(position, Math.max(0, audio.duration - .1)); await audio.play();
    } catch { $('listen-status').textContent = '浏览器未能播放，请检查MP3链接。'; }
  }));
  $('training').addEventListener('toggle', draw);
  new ResizeObserver(draw).observe($('reward-chart'));
  render();
} catch (error) {
  $('model-note').textContent = '实验记录加载失败，未显示替代结果。'; console.error(error);
}
