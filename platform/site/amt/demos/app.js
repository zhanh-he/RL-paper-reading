const $ = selector => document.querySelector(selector);
const text = (tag, value, className) => {
  const node = document.createElement(tag);
  node.textContent = value;
  if (className) node.className = className;
  return node;
};
const colors = ['#e3484f', '#d47b16', '#008a93', '#315fd2', '#7653a5', '#176b57'];
const stateLabels = { planned: '待执行', queued: '已排队', running: '运行中', measured: '实测', failed: '失败' };
const isAudit = () => experiment?.kind === 'evaluation';
const runPoints = () => isAudit() ? experiment.conditions : experiment?.stages || [];
const pointKey = point => isAudit() ? point.id : String(point.step);
let manifest, experiment, sample, stage;
let hoverStep = null;
const mediaURL = value => new URL(`../../${value}`, location.href).href;
const receiptURL = value => `https://github.com/zhanh-he/RL-paper-reading/blob/main/${value}`;

function link(label, href) {
  const node = text('a', label, 'source-link');
  node.href = href;
  return node;
}

function selectOptions(select, options, selected) {
  select.replaceChildren(...options.map(([id, title]) => {
    const option = new Option(title, id);
    option.selected = id === selected;
    return option;
  }));
  select.disabled = options.length === 0;
  if (!options.length) select.append(new Option('暂无样本', ''));
}

function setTab(id) {
  if (!['experiments', 'datasets', 'models', 'rewards'].includes(id)) id = 'experiments';
  for (const tab of document.querySelectorAll('[data-tab]')) {
    const selected = tab.dataset.tab === id;
    tab.setAttribute('aria-selected', String(selected));
    tab.tabIndex = selected ? 0 : -1;
    $(`#panel-${tab.dataset.tab}`).hidden = !selected;
  }
  for (const audio of document.querySelectorAll('[role=tabpanel][hidden] audio')) audio.pause();
  if (location.hash !== `#${id}`) history.replaceState(null, '', `#${id}`);
  if (id === 'experiments') requestAnimationFrame(drawCharts);
}

for (const tab of document.querySelectorAll('[data-tab]')) {
  tab.addEventListener('click', () => setTab(tab.dataset.tab));
  tab.addEventListener('keydown', event => {
    const tabs = [...document.querySelectorAll('[data-tab]')];
    let index = tabs.indexOf(tab);
    if (event.key === 'ArrowRight') index = (index + 1) % tabs.length;
    else if (event.key === 'ArrowLeft') index = (index + tabs.length - 1) % tabs.length;
    else if (event.key === 'Home') index = 0;
    else if (event.key === 'End') index = tabs.length - 1;
    else return;
    event.preventDefault();
    setTab(tabs[index].dataset.tab);
    tabs[index].focus();
  });
}
window.addEventListener('hashchange', () => setTab(location.hash.slice(1)));

function pianoRoll(canvas, asset, duration, notes) {
  const ctx = canvas.getContext('2d'), w = canvas.width, h = canvas.height;
  const pitches = notes.map(note => note.pitch);
  const low = Math.min(48, ...pitches) - 2, high = Math.max(72, ...pitches) + 2;
  const x0 = 45, y0 = 14, width = w - x0 - 12, height = h - y0 - 32;
  ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, w, h);
  ctx.font = '12px system-ui';
  for (let pitch = low; pitch <= high; pitch++) {
    const y = y0 + (high - pitch) / (high - low + 1) * height;
    ctx.fillStyle = [1, 3, 6, 8, 10].includes(pitch % 12) ? '#f1f3f2' : '#fff';
    ctx.fillRect(x0, y, width, height / (high - low + 1));
    if (pitch % 12 === 0) { ctx.fillStyle = '#5f6b66'; ctx.fillText(`C${pitch / 12 - 1}`, 8, y + 10); }
  }
  for (let i = 0; i <= 4; i++) {
    const x = x0 + i / 4 * width;
    ctx.strokeStyle = '#d7ddd9'; ctx.beginPath(); ctx.moveTo(x, y0); ctx.lineTo(x, y0 + height); ctx.stroke();
    ctx.fillStyle = '#5f6b66'; ctx.fillText(`${(duration * i / 4).toFixed(1)}s`, Math.min(x, w - 38), h - 8);
  }
  for (const note of asset.notes || []) {
    const y = y0 + (high - note.pitch) / (high - low + 1) * height;
    ctx.fillStyle = colors[note.track % colors.length];
    ctx.fillRect(x0 + note.start / duration * width, y, Math.max(2, (note.end - note.start) / duration * width), Math.max(2, height / (high - low + 1) - 1));
  }
}

function renderMedia() {
  for (const audio of $('#media-grid').querySelectorAll('audio')) audio.pause();
  const baseline = runPoints()[0];
  const entries = isAudit() ? [
    ['Input · 输入', sample?.input, '输入未公开'],
    ['GT · 参考音符', sample?.reference, '无准确参考标签'],
    ['Original · 原版输出', baseline?.outputs?.[sample?.id], '尚无原版推理结果'],
    [stage?.label || 'Condition', stage?.outputs?.[sample?.id], stage?.error || experiment?.error || '尚无该条件输出'],
  ] : [
    ['Input · 原始音频', sample?.input, '输入待提供'],
    ['GT · 参考 MIDI', sample?.reference, '参考待提供 / 可选'],
    ['Frozen · Baseline', baseline?.outputs?.[sample?.id], '基线待运行'],
    [stage ? `${stage.step} steps · ${experiment.method}` : 'Post-training', stage?.outputs?.[sample?.id], '训练输出待提供'],
  ];
  const allNotes = entries.flatMap(([, asset]) => asset?.notes || []);
  $('#media-grid').replaceChildren(...entries.map(([title, asset, pending]) => {
    const article = text('article', '', 'amt-media');
    article.append(text('h4', title));
    if (Array.isArray(asset?.notes)) {
      const zoom = text('button', '', 'amt-preview'); zoom.type = 'button'; zoom.title = `放大 ${title} MIDI`;
      zoom.setAttribute('aria-label', zoom.title);
      const canvas = document.createElement('canvas'); canvas.width = 700; canvas.height = 390;
      canvas.setAttribute('role', 'img'); canvas.setAttribute('aria-label', `${title} piano roll`);
      pianoRoll(canvas, asset, sample.duration, allNotes);
      zoom.append(canvas, text('span', '放大 ↗', 'zoom-hint'));
      zoom.addEventListener('click', () => {
        $('#roll-dialog-title').textContent = title;
        pianoRoll($('#large-roll'), asset, sample.duration, allNotes);
        $('#roll-dialog').showModal();
      });
      article.append(zoom);
      const tracks = [...new Set(asset.notes.map(note => note.track))];
      const legend = text('small', '');
      for (const track of tracks) {
        const label = text('span', `${asset.tracks?.[track] || `Track ${track + 1}`}  `);
        label.style.color = colors[track % colors.length]; legend.append(label);
      }
      article.append(legend);
    } else article.append(text('div', asset?.audio ? 'Audio' : pending, 'amt-preview'));
    if (asset?.audio) {
      const audio = document.createElement('audio'); audio.controls = true; audio.preload = 'metadata'; audio.src = mediaURL(asset.audio);
      audio.setAttribute('aria-label', title); article.append(audio);
    } else if (asset) article.append(text('p', '试听待提供'));
    if (asset?.midi) { const download = link('MIDI ↓', mediaURL(asset.midi)); download.download = ''; article.append(download); }
    return article;
  }));
  $('#sample-context').textContent = sample?.note || (experiment ? '暂无公开回放样本。' : '实验任务、模型、数据划分与 reward 尚未确定。');
}

function formatMetric(value, baseline) {
  if (value == null) return '—';
  const score = `${(value * 100).toFixed(1)}%`;
  if (baseline == null) return score;
  const delta = (value - baseline) * 100;
  return `${score} (${delta >= 0 ? '+' : ''}${delta.toFixed(1)} pp)`;
}

function renderMetrics() {
  if (isAudit()) {
    const definitions = experiment.metric_definitions;
    $('#metric-head').replaceChildren(...['Condition / Status', 'Cohort / N', ...definitions.map(item => item.label)].map(value => text('th', value)));
    $('#metrics-body').replaceChildren(...experiment.conditions.map(item => {
      const row = document.createElement('tr'); row.dataset.selected = String(item === stage);
      const result = item.evaluation;
      const values = [item.label + ' · ' + stateLabels[item.status], result ? `${result.cohort} / ${result.n}` : item.error || '未评估', ...definitions.map(metric => {
        const value = result?.values?.[metric.id];
        return value == null ? '—' : metric.unit === 'fraction' ? `${(value*100).toFixed(2)}%` : value.toFixed(2);
      })];
      row.append(...values.map(value => text('td', value))); return row;
    }));
    return;
  }
  $('#metric-head').replaceChildren(...['Step', 'Cohort / N', 'Frame F1', 'Onset F1', 'Onset + offset F1'].map(value => text('th', value)));
  const rows = experiment?.stages || [];
  const baseline = rows.find(item => item.step === 0)?.evaluation;
  $('#metrics-body').replaceChildren(...(rows.length ? rows.map(item => {
    const tr = document.createElement('tr'); tr.dataset.selected = String(item.step === stage?.step);
    const result = item.evaluation;
    const sameCohort = result && baseline && result.cohort === baseline.cohort && result.n === baseline.n;
    for (const value of [item.step, result ? `${result.cohort} / ${result.n}` : '待评估', ...['frame', 'onset', 'onset_offset'].map(metric => formatMetric(result?.[metric], sameCohort && item.step !== 0 ? baseline?.[metric] : null))]) tr.append(text('td', value));
    return tr;
  }) : [(() => { const row = document.createElement('tr'); const cell = text('td', '暂无评估记录'); cell.colSpan = 5; row.append(cell); return row; })()]));
}

function chooseStage(item) {
  stage = item; hoverStep = null;
  for (const button of $('#steps').querySelectorAll('button')) button.setAttribute('aria-pressed', String(button.dataset.key === pointKey(item)));
  renderMedia(); renderMetrics(); drawCharts();
}

function chooseExperiment(id) {
  experiment = manifest.experiments.find(item => item.id === id);
  stage = isAudit() ? experiment.conditions[0] : experiment?.stages.find(item => item.step > 0) || experiment?.stages[0];
  sample = experiment?.samples[0]; hoverStep = null;
  $('#page-status').textContent = experiment ? stateLabels[experiment.status] : '实验待配置';
  $('#page-status').className = `state ${experiment?.status === 'measured' ? 'measured' : 'pending'}`;
  $('#experiment-title').textContent = experiment?.title || '下一项实验待配置';
  if (experiment) {
    const params = new URLSearchParams(location.search); params.set('run', id);
    history.replaceState(null, '', `${location.pathname}?${params}${location.hash}`);
    $('#run-protocol').replaceChildren(...['task', 'model', 'method', 'reward'].map(key => { const span = text('span', `${key} `); span.append(text('strong', experiment[key] || '待定')); return span; }));
  }
  selectOptions($('#sample-select'), (experiment?.samples || []).map(item => [item.id, item.title]), sample?.id);
  $('#run-receipt').hidden = !experiment?.receipt;
  if (experiment?.receipt) $('#run-receipt').href = receiptURL(experiment.receipt);
  $('#evaluation-protocol').textContent = experiment?.evaluation_protocol || '待确定时间容差、声部/乐器匹配、聚合方式与测试集合。';
  $('#training-section').hidden = isAudit();
  $('#metrics-title').textContent = isAudit() ? '冻结评测 · 条件对照' : 'Checkpoint · 独立评估';
  $('#metrics-context').textContent = isAudit() ? '原始测量值 · 非训练得分' : '分数（Δ vs baseline）';
  $('#replay-title').textContent = isAudit() ? sample ? '同一样本 · 条件与参考' : '条件对照' : '同一首歌 · MIDI 与试听';
  const noMedia = isAudit() && !sample;
  $('#media-grid').hidden = noMedia;
  $('#sample-context').hidden = noMedia;
  $('#columns').parentElement.hidden = noMedia;
  $('#steps').setAttribute('aria-label', isAudit() ? 'Evaluation conditions' : 'Optimizer checkpoints');
  $('#trace-context').textContent = experiment?.trace_note || '等待训练日志';
  const checkpoints = runPoints();
  $('#steps').replaceChildren(...checkpoints.map(item => {
    const button = text('button', '', `stage-item ${item.status === 'measured' ? 'available' : ''}`); button.type = 'button'; button.dataset.step = item.step; button.dataset.key = pointKey(item);
    const point = experiment.trace.find(point => point.step === item.step);
    const value = key => Number.isFinite(point?.[key]) ? point[key].toFixed(4) : '—';
    button.title = stateLabels[item.status];
    button.append(text('strong', isAudit() ? item.label : `${item.step} steps`), text('small', isAudit() ? stateLabels[item.status] : item.status === 'planned' ? stateLabels[item.status] : `R ${value('reward')} · KL ${value('kl')}`));
    button.setAttribute('aria-pressed', String(item === stage));
    button.addEventListener('click', () => chooseStage(item)); return button;
  }));
  if (!checkpoints.length) $('#steps').append(text('span', 'Checkpoint 待配置', 'amt-inline-empty'));
  renderMedia(); renderMetrics(); drawCharts();
}

function drawCharts() {
  for (const metric of ['reward', 'kl']) {
    const canvas = $(`#${metric}-chart`), empty = canvas.nextElementSibling;
    const points = (experiment?.trace || []).filter(point => Number.isFinite(point[metric]));
    canvas.hidden = !points.length; empty.hidden = !!points.length;
    $(`#${metric}-value`).textContent = '—';
    if (!points.length || !canvas.clientWidth) continue;
    const w = canvas.clientWidth, h = canvas.clientHeight, dpr = devicePixelRatio || 1;
    canvas.width = w * dpr; canvas.height = h * dpr;
    const ctx = canvas.getContext('2d'); ctx.scale(dpr, dpr);
    const left = 48, top = 14, right = w - 16, bottom = h - 26;
    const last = Math.max(1, ...points.map(point => point.step), ...(experiment?.stages || []).map(item => item.step));
    const min = Math.min(...points.map(point => point[metric])), max = Math.max(...points.map(point => point[metric]));
    const pad = Math.max((max - min) * .12, Math.abs(max) * .01, .001);
    const x = step => left + step / last * (right - left), y = value => bottom - (value - min + pad) / (max - min + 2 * pad) * (bottom - top);
    ctx.font = '10px system-ui'; ctx.fillStyle = '#5f6b66';
    for (let i = 0; i <= 2; i++) {
      const v = min + (max - min) * i / 2, yy = y(v);
      ctx.fillText(v.toFixed(3), 2, yy + 3); ctx.strokeStyle = '#e1e5e2'; ctx.beginPath(); ctx.moveTo(left, yy); ctx.lineTo(right, yy); ctx.stroke();
    }
    ctx.fillText('0', left, h - 7); ctx.fillText(`${last} steps`, right - 48, h - 7);
    ctx.strokeStyle = metric === 'reward' ? '#176b57' : '#245ea8'; ctx.lineWidth = 2; ctx.beginPath();
    points.forEach((point, i) => i ? ctx.lineTo(x(point.step), y(point[metric])) : ctx.moveTo(x(point.step), y(point[metric]))); ctx.stroke();
    const selected = hoverStep ?? stage?.step;
    if (selected != null) {
      const nearest = points.reduce((a, b) => Math.abs(a.step - selected) <= Math.abs(b.step - selected) ? a : b);
      ctx.strokeStyle = '#69756f'; ctx.lineWidth = 1; ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.moveTo(x(selected), top); ctx.lineTo(x(selected), bottom); ctx.stroke(); ctx.setLineDash([]);
      ctx.beginPath(); ctx.arc(x(nearest.step), y(nearest[metric]), 3, 0, Math.PI * 2); ctx.fillStyle = metric === 'reward' ? '#176b57' : '#245ea8'; ctx.fill();
      $(`#${metric}-value`).textContent = `${nearest.step} steps · ${nearest[metric].toFixed(4)}`;
    }
  }
}

for (const metric of ['reward', 'kl']) {
  const canvas = $(`#${metric}-chart`);
  canvas.addEventListener('pointermove', event => {
    const points = experiment?.trace || [];
    const max = Math.max(1, ...points.map(point => point.step), ...(experiment?.stages || []).map(item => item.step));
    hoverStep = Math.max(0, Math.min(max, (event.offsetX - 48) / (canvas.clientWidth - 64) * max)); drawCharts();
  });
  canvas.addEventListener('pointerleave', () => { hoverStep = null; drawCharts(); });
}
$('#experiment-select').addEventListener('change', event => chooseExperiment(event.target.value));
$('#sample-select').addEventListener('change', event => { sample = experiment.samples.find(item => item.id === event.target.value); renderMedia(); });
$('#columns').addEventListener('change', event => $('#media-grid').style.setProperty('--columns', event.target.value));
$('#close-roll').addEventListener('click', () => $('#roll-dialog').close());
$('#roll-dialog').addEventListener('click', event => { if (event.target === event.currentTarget) $('#roll-dialog').close(); });
window.addEventListener('resize', drawCharts);

function renderReference(kind, fields) {
  const records = manifest[kind];
  if (!records.length) return;
  $(`#${kind}-body`).replaceChildren(...records.map(record => {
    const row = document.createElement('tr');
    for (const field of fields) row.append(text('td', record[field] || '待配置'));
    const source = document.createElement('td'); source.append(link(record.source_label || '来源 ↗', record.source)); row.append(source);
    return row;
  }));
  const badge = $(`#panel-${kind} .state`); badge.textContent = `${records.length} 项`; badge.className = 'state diagnostic';
}

setTab(location.hash.slice(1));
renderMedia();
try {
  const response = await fetch('./data/experiments.json', { cache: 'no-store' });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  manifest = await response.json();
  $('#manifest-date').textContent = `更新 ${manifest.updated_at}`;
  if (manifest.experiments.length) {
    selectOptions($('#experiment-select'), manifest.experiments.map(item => [item.id, item.title]), null);
    const requested = new URLSearchParams(location.search).get('run');
    const id = manifest.experiments.some(item => item.id === requested) ? requested : manifest.experiments[0].id;
    $('#experiment-select').value = id; chooseExperiment(id);
  } else $('#page-status').textContent = '实验待配置';
  renderReference('datasets', ['name', 'labels', 'split']);
  renderReference('models', ['name', 'interface', 'checkpoint']);
  renderReference('rewards', ['name', 'definition', 'failure_case']);
  if (manifest.reported_results?.length) {
    $('#paper-comparison').hidden = false;
    $('#reported-results-body').replaceChildren(...manifest.reported_results.map(record => {
      const row = document.createElement('tr');
      row.append(...[record.name, record.rwc.toFixed(2), record.rock.toFixed(2), record.output].map(value => text('td', value)));
      const source = document.createElement('td'); source.append(link('Table 17 ↗', record.source)); row.append(source);
      return row;
    }));
  }
} catch (error) {
  $('#page-status').textContent = '实验清单加载失败';
  $('#sample-context').textContent = `无法读取实验记录：${error.message}`;
}
if (window.lucide) window.lucide.createIcons();
