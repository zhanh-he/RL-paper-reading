const files = {
  coverage: './vocal-lada-coverage-run.json',
  beat_v2: './vocal-lada-beat_v2-run.json',
  beat_v5: './vocal-lada-beat_v5-run.json',
  richness_v0: './vocal-lada-richness_v0-run.json',
  combined: './vocal-lada-run.json',
  guarded: './vocal-lada-guarded-run.json',
};
const names = {
  coverage: 'Coverage only',
  beat_v2: 'Beat-v2 only',
  beat_v5: 'Beat-v5 only',
  richness_v0: 'Richness-v0 proxy',
  combined: 'Proxy blend (no Beat-v2)',
  guarded: 'Beat-v2 + Coverage guard',
};
const steps = [0, 5, 50, 100, 150, 200, 300];
const root = document.getElementById('view-vocal');
const el = (selector) => root.querySelector(selector);
const fmt = (value, digits = 3) => Number.isFinite(value) ? value.toFixed(digits) : '—';
const stageMetric = (value) => Number.isFinite(value)
  ? value.toFixed(Math.abs(value) < 0.001 ? 6 : 3) : '—';
let arm = 'beat_v2';
let selectedStep = 100;
let run;

function textNode(tag, className, value) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  node.textContent = value;
  return node;
}

function panel(stage, side) {
  const node = document.createElement('article');
  node.className = 'compare-panel';
  node.append(textNode('span', 'source-label', side === 'a' ? 'A / FROZEN' : `B / ${names[arm].toUpperCase()}`));
  node.append(textNode('h3', '', side === 'a' ? 'Baseline · 0 updates' : `GRPO · ${stage.step} updates`));
  const wave = document.createElement('div');
  wave.className = 'wave-visual';
  const waveImage = document.createElement('img');
  waveImage.src = stage.audio.replace('/audio/', '/visuals/').replace(/\.wav$/, '_wave.png');
  waveImage.alt = `${side === 'a' ? 'Baseline' : `GRPO ${stage.step}`} mix waveform`;
  wave.append(waveImage);
  node.append(wave);
  const spectrum = document.createElement('figure');
  spectrum.className = 'spectrum-visual';
  const spectrumImage = document.createElement('img');
  spectrumImage.src = stage.spectrum;
  spectrumImage.alt = `${side === 'a' ? 'Baseline' : `GRPO ${stage.step}`} accompaniment spectrogram`;
  spectrum.append(spectrumImage, textNode('figcaption', '', '伴奏频谱 · 0–24 kHz'));
  node.append(spectrum);
  node.append(textNode('span', 'source-label', '人声 + 伴奏'));
  const mix = document.createElement('audio');
  mix.controls = true;
  mix.preload = 'metadata';
  mix.src = stage.audio;
  mix.dataset.ladaSide = side;
  node.append(mix);
  node.append(textNode('span', 'source-label', '伴奏独听'));
  const accompaniment = document.createElement('audio');
  accompaniment.controls = true;
  accompaniment.preload = 'metadata';
  accompaniment.src = stage.accomp_audio;
  node.append(accompaniment);
  node.append(textNode('small', '', stage.same_as_baseline ? '与 baseline 音频逐字节相同' : `Beat-v2 F1 ${fmt(stage.beat_v2?.score)} · RMS coverage ${fmt(stage.rms_coverage)}`));
  return node;
}

function trainingChart(selector, key, color) {
  const canvas = el(selector);
  const tooltip = textNode('div', 'training-chart-tooltip', '');
  tooltip.hidden = true;
  canvas.parentElement.append(tooltip);
  let points = [];
  let checkpoint = 0;
  const height = 190;
  const margin = { left: 47, right: 13, top: 15, bottom: 26 };
  const transform = key === 'sampled_kl' ? (value) => Math.asinh(value / 0.05) : (value) => value;

  function draw() {
    const width = canvas.getBoundingClientRect().width;
    if (width < 80 || !points.length) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    const context = canvas.getContext('2d');
    context.scale(ratio, ratio);
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    const maxStep = points.at(-1).step;
    const values = points.map((point) => point[key]).filter(Number.isFinite);
    if (!values.length) return;
    const low = Math.min(...values);
    const high = Math.max(...values);
    const pad = Math.max((high - low) * 0.12, 0.015);
    const bottom = key === 'sampled_kl' ? 0 : low - pad;
    const top = key === 'sampled_kl'
      ? ([0.03, 0.1, 0.3, 1, 3, 10, 30, 100].find((tick) => tick >= high * 1.05) || high * 1.1)
      : high + pad;
    const x = (step) => margin.left + plotWidth * step / maxStep;
    const y = (value) => margin.top + plotHeight * (transform(top) - transform(value)) / (transform(top) - transform(bottom));
    const ticks = key === 'sampled_kl'
      ? [0, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100].filter((tick) => tick <= top)
      : [bottom, (bottom + top) / 2, top];
    context.font = '11px system-ui, sans-serif';
    context.fillStyle = '#68726d';
    context.textAlign = 'right';
    let previousTickY = Infinity;
    for (const tick of ticks) {
      const tickY = y(tick);
      if (previousTickY - tickY < 18) continue;
      previousTickY = tickY;
      context.strokeStyle = '#dbe2dd';
      context.lineWidth = 1;
      context.beginPath();
      context.moveTo(margin.left, tickY);
      context.lineTo(width - margin.right, tickY);
      context.stroke();
      context.fillText(Math.abs(tick) < 1 ? tick.toFixed(tick === 0 ? 0 : 2) : tick.toFixed(1), margin.left - 6, tickY + 4);
    }
    context.textAlign = 'center';
    for (const step of [0, Math.round(maxStep / 2), maxStep]) context.fillText(String(step), x(step), height - 5);
    const trace = (field, alpha, lineWidth) => {
      context.beginPath();
      context.strokeStyle = color;
      context.globalAlpha = alpha;
      context.lineWidth = lineWidth;
      context.lineJoin = 'round';
      points.forEach((point, index) => {
        const pointY = y(point[field]);
        if (index) context.lineTo(x(point.step), pointY); else context.moveTo(x(point.step), pointY);
      });
      context.stroke();
      context.globalAlpha = 1;
    };
    trace(key, 0.18, 1);
    trace('window', 1, 2.5);
    for (const stage of run.stages) {
      if (stage.step === 0 || stage.step > maxStep) continue;
      const point = points.find((item) => item.step === stage.step);
      if (!point) continue;
      context.beginPath();
      context.arc(x(stage.step), y(point.window), stage.step === checkpoint ? 5 : 3.5, 0, Math.PI * 2);
      context.fillStyle = '#fff';
      context.fill();
      context.strokeStyle = color;
      context.lineWidth = 2;
      context.stroke();
    }
    if (checkpoint > 0 && checkpoint <= maxStep) {
      context.beginPath();
      context.setLineDash([3, 4]);
      context.strokeStyle = '#63716a';
      context.moveTo(x(checkpoint), margin.top);
      context.lineTo(x(checkpoint), height - margin.bottom);
      context.stroke();
      context.setLineDash([]);
    }
  }

  canvas.addEventListener('pointermove', (event) => {
    const bounds = canvas.getBoundingClientRect();
    const plotWidth = bounds.width - margin.left - margin.right;
    if (!points.length || plotWidth <= 0) return;
    const step = Math.round(Math.max(0, Math.min(1, (event.clientX - bounds.left - margin.left) / plotWidth)) * points.at(-1).step);
    const point = points.reduce((nearest, item) => Math.abs(item.step - step) < Math.abs(nearest.step - step) ? item : nearest);
    tooltip.textContent = `${point.step} 步 · 10 步均值 ${fmt(point.window, 4)}\n本步原始值 ${fmt(point[key], 4)}`;
    tooltip.style.left = `${Math.max(80, Math.min(bounds.width - 80, event.clientX - bounds.left))}px`;
    tooltip.style.top = `${canvas.offsetTop + 8}px`;
    tooltip.hidden = false;
  });
  canvas.addEventListener('pointerleave', () => { tooltip.hidden = true; });
  new ResizeObserver(draw).observe(canvas.parentElement);
  return {
    set(raw, selected) {
      const valid = raw.filter((point) => Number.isFinite(point[key]));
      points = valid.map((point, index) => ({
        ...point,
        window: valid.slice(Math.max(0, index - 9), index + 1).reduce((sum, entry) => sum + entry[key], 0) / Math.min(index + 1, 10),
      }));
      checkpoint = selected;
      tooltip.hidden = true;
      draw();
      return points;
    },
  };
}

const rewardChart = trainingChart('#lada-reward-chart', 'reward', '#08745d');
const klChart = trainingChart('#lada-kl-chart', 'sampled_kl', '#b36b24');

function render() {
  const baseline = run.stages.find((stage) => stage.step === 0);
  const selected = run.stages.find((stage) => stage.step === selectedStep && stage.audio) || baseline;
  selectedStep = selected.step;
  el('#lada-source').src = run.source_audio;
  el('#lada-protocol-title').textContent = `LaDA-Band · ${names[arm]}`;
  el('#lada-protocol').textContent = `相同 Emma 人声、提示、8 步去噪、group 2、LR ${run.config.lr}、固定评估 seed ${run.config.eval_seed}。仅 LoRA 输出投影更新；${arm === 'beat_v2' ? '训练直接使用原 Madmom Beat-v2 F1。' : arm === 'beat_v5' ? '训练使用原项目 Beat-v5 onset-grid 分数；低 confidence 时可 abstain，12 秒回放另算独立 Beat-v2。' : arm === 'richness_v0' ? '训练优化多频段活动与时间变化的 Richness-v0 代理，并使用音调性与人声相对响度约束；它不会判断和声是否正确。' : arm === 'coverage' ? '训练仅优化 40 ms RMS coverage。' : arm === 'guarded' ? '训练使用原 Beat-v2 + 饱和 coverage + 响度/平坦度约束。' : '训练使用 coverage、能量起音和频带占用代理；并非原 Beat-v2。'}`;
  el('#lada-training-arm').textContent = names[arm];
  el('#lada-training-extent').textContent = `1–${run.train_curve.at(-1)?.step || 0} 步 · 每步 2 条 6 秒采样`;
  const rewardPoints = rewardChart.set(run.train_curve, selectedStep);
  const klPoints = klChart.set(run.train_curve, selectedStep);
  el('#lada-reward-value').textContent = rewardPoints.length ? `${fmt(rewardPoints[9]?.window ?? rewardPoints.at(-1).window)} → ${fmt(rewardPoints.at(-1).window)}` : '无记录';
  el('#lada-kl-value').textContent = klPoints.length ? `${fmt(klPoints[9]?.window ?? klPoints.at(-1).window)} → ${fmt(klPoints.at(-1).window)}` : '无记录';
  const klPeak = klPoints.reduce((peak, point) => !peak || point.sampled_kl > peak.sampled_kl ? point : peak, null);
  el('#lada-training-note').textContent = `细线为每步原始值，粗线为过去最多 10 步的滑动均值；圆点是有固定重放的 checkpoint，虚线是当前试听步数。KL 纵轴为 asinh 非线性刻度，单步峰值 ${klPeak ? `${fmt(klPeak.sampled_kl, 3)}（${klPeak.step} 步）` : '未记录'}；每图刻度独立。训练候选只取前 6 秒，下方 A/B 是固定 seed 的 12 秒重放。`;
  const rail = el('#lada-stage-rail');
  rail.replaceChildren();
  for (const step of steps) {
    const stage = run.stages.find((item) => item.step === step && item.audio);
    const item = document.createElement(stage ? 'button' : 'div');
    item.className = `stage-item${stage ? ' available' : ''}`;
    item.append(textNode('strong', '', step === 0 ? 'Baseline' : `${step} steps`));
    const metric = run.train_curve.find((point) => point.step === step);
    const stageValue = step === 0 ? '冻结起点' : metric
      ? `训练 R ${stageMetric(metric.reward)} · KL ${stageMetric(metric.sampled_kl)}` : '无训练值';
    item.append(textNode('span', `state ${stage ? 'measured' : 'pending'}`, stage ? stageValue : '未发布'));
    if (stage) {
      item.type = 'button';
      item.setAttribute('aria-pressed', String(step === selectedStep));
      item.addEventListener('click', () => { selectedStep = step; render(); });
    }
    rail.append(item);
  }
  el('#lada-stage-context').textContent = selected.step === 0
    ? '冻结模型基线；所有 reward 组使用逐字节相同的 step-0 WAV。'
    : `${selected.step} 步固定 12 秒重放；在线更新只使用前 6 秒。${selected.same_as_baseline ? '本步音频与基线逐字节相同。' : '音频按模型原始电平播放，响度变化本身需要检查。'}`;
  el('#lada-compare').replaceChildren(panel(baseline, 'a'), panel(selected, 'b'));
  el('#lada-metric-head').textContent = `${selected.step} steps`;
  const metrics = [
    ['本臂目标 · 固定 12 s', baseline.reward, selected.reward, 3],
    ['Beat-v2 F1', baseline.beat_v2?.score, selected.beat_v2?.score, 3],
    ['RMS coverage', baseline.rms_coverage, selected.rms_coverage, 3],
    ['STFT coverage', baseline.beat_v2?.coverage_stft, selected.beat_v2?.coverage_stft, 3],
    ['伴奏/人声 RMS dB', baseline.beat_v2?.acc_to_vocal_rms_db, selected.beat_v2?.acc_to_vocal_rms_db, 1],
    ['伴奏峰值', baseline.beat_v2?.stereo_peak, selected.beat_v2?.stereo_peak, 3],
    ['削波占比', baseline.beat_v2?.stereo_clipping_fraction, selected.beat_v2?.stereo_clipping_fraction, 5],
  ];
  if (arm === 'beat_v5') {
    metrics.splice(1, 0, ['Beat-v5 confidence', baseline.beat_v5_confidence, selected.beat_v5_confidence, 3]);
  }
  if (arm === 'richness_v0') {
    metrics.splice(1, 0,
      ['频段活动', baseline.layer_activity, selected.layer_activity, 3],
      ['时间变化', baseline.layer_movement, selected.layer_movement, 3],
      ['音调性门控', baseline.tonality_gate, selected.tonality_gate, 3],
      ['响度惩罚', baseline.loudness_guard_penalty, selected.loudness_guard_penalty, 3]);
  }
  const body = el('#lada-metrics');
  body.replaceChildren();
  for (const [name, before, after, digits] of metrics) {
    const row = document.createElement('tr');
    [name, fmt(before, digits), fmt(after, digits), Number.isFinite(before) && Number.isFinite(after) ? `${after - before >= 0 ? '+' : ''}${fmt(after - before, digits)}` : '—']
      .forEach((value) => row.append(textNode('td', '', value)));
    body.append(row);
  }
  el('#lada-interpretation').textContent = `${names[arm]} · 同一首 Emma 原创旋律的固定重放。${selected.step ? `本步 Beat-v2 F1 ${fmt(baseline.beat_v2?.score)} → ${fmt(selected.beat_v2?.score)}，STFT coverage ${fmt(baseline.beat_v2?.coverage_stft)} → ${fmt(selected.beat_v2?.coverage_stft)}。` : ''} 这些客观分数不能代替盲听；未见歌曲和独立歌手尚未验证。`;
  el('#lada-listen-b').textContent = `B · ${selected.step} steps`;
  for (const button of el('#lada-arms').querySelectorAll('button')) button.setAttribute('aria-pressed', String(button.dataset.arm === arm));
}

async function loadArm(nextArm) {
  try {
    const response = await fetch(files[nextArm], { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    run = await response.json();
    arm = nextArm;
    if (!run.stages.some((stage) => stage.step === selectedStep && stage.audio)) selectedStep = run.stages.some((stage) => stage.step === 100 && stage.audio) ? 100 : run.stages.at(-1)?.step || 0;
    render();
  } catch {
    el('#lada-stage-context').textContent = '该 reward 组尚无可验证的音频与指标。';
  }
}

for (const key of Object.keys(files)) {
  const button = textNode('button', '', names[key]);
  button.type = 'button';
  button.dataset.arm = key;
  button.addEventListener('click', () => loadArm(key));
  el('#lada-arms').append(button);
}
for (const [side, selector] of [['a', '#lada-listen-a'], ['b', '#lada-listen-b']]) {
  el(selector).addEventListener('click', () => {
    const target = el(`[data-lada-side="${side}"]`);
    const other = el(`[data-lada-side="${side === 'a' ? 'b' : 'a'}"]`);
    const time = other.currentTime;
    other.pause();
    const play = () => { target.currentTime = Math.min(time, Math.max(0, target.duration - 0.1)); target.play().catch(() => { el('#lada-listen-status').textContent = '请点击播放器开始播放'; }); };
    if (target.readyState >= 1) play(); else target.addEventListener('loadedmetadata', play, { once: true });
    el('#lada-listen-status').textContent = side === 'a' ? 'A · Baseline' : `B · ${selectedStep} steps`;
  });
}
loadArm(arm);
