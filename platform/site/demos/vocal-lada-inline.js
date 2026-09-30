const files = {
  combined: './vocal-lada-run.json',
  coverage: './vocal-lada-coverage-run.json',
  beat_v2: './vocal-lada-beat_v2-run.json',
  guarded: './vocal-lada-guarded-run.json',
};
const names = {
  combined: 'Combined proxy',
  coverage: 'Coverage only',
  beat_v2: 'Beat-v2 only',
  guarded: 'Beat + Coverage',
};
const steps = [0, 5, 50, 100, 150, 200, 300];
const root = document.getElementById('view-vocal');
const el = (selector) => root.querySelector(selector);
const fmt = (value, digits = 3) => Number.isFinite(value) ? value.toFixed(digits) : '—';
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

function drawCurve(points) {
  const canvas = el('#lada-curve');
  const context = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;
  context.clearRect(0, 0, width, height);
  context.strokeStyle = '#b9c9c0';
  for (let i = 0; i <= 4; i++) {
    const y = 18 + i * (height - 40) / 4;
    context.beginPath();
    context.moveTo(34, y);
    context.lineTo(width - 12, y);
    context.stroke();
  }
  if (!points?.length) return;
  const maxStep = Math.max(...points.map((point) => point.step), 1);
  for (const [key, color] of [['reward', '#16876d'], ['rms_coverage', '#bf7950']]) {
    context.beginPath();
    context.strokeStyle = color;
    context.lineWidth = 2;
    points.forEach((point, index) => {
      const x = 34 + point.step / maxStep * (width - 50);
      const y = height - 20 - Math.max(0, Math.min(1, point[key])) * (height - 40);
      if (index) context.lineTo(x, y); else context.moveTo(x, y);
    });
    context.stroke();
  }
  context.fillStyle = '#566a60';
  context.font = '12px sans-serif';
  context.fillText('0', 12, height - 15);
  context.fillText('1', 12, 22);
  context.fillText(`${maxStep} steps`, width - 80, height - 5);
}

function render() {
  const baseline = run.stages.find((stage) => stage.step === 0);
  const selected = run.stages.find((stage) => stage.step === selectedStep && stage.audio) || baseline;
  selectedStep = selected.step;
  el('#lada-source').src = run.source_audio;
  el('#lada-protocol-title').textContent = `LaDA-Band · ${names[arm]}`;
  el('#lada-protocol').textContent = `相同 Emma 人声、提示、8 步去噪、group 2、LR ${run.config.lr}、固定评估 seed ${run.config.eval_seed}。仅 LoRA 输出投影更新；${arm === 'beat_v2' ? '训练直接使用原 Madmom Beat-v2 F1。' : arm === 'coverage' ? '训练仅优化 40 ms RMS coverage。' : arm === 'guarded' ? '训练使用原 Beat-v2 + 饱和 coverage + 响度/平坦度约束。' : '训练使用 coverage、能量起音和频带占用代理；并非原 Beat-v2。'}`;
  el('#lada-curve-value').textContent = fmt(run.train_curve.at(-1)?.reward);
  drawCurve(run.train_curve);
  const rail = el('#lada-stage-rail');
  rail.replaceChildren();
  for (const step of steps) {
    const stage = run.stages.find((item) => item.step === step && item.audio);
    const item = document.createElement(stage ? 'button' : 'div');
    item.className = `stage-item${stage ? ' available' : ''}`;
    item.append(textNode('strong', '', step === 0 ? 'Baseline' : `${step} steps`));
    item.append(textNode('span', `state ${stage ? 'measured' : 'pending'}`, stage ? '实测' : '未发布'));
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
    ['Beat-v2 F1', baseline.beat_v2?.score, selected.beat_v2?.score, 3],
    ['RMS coverage', baseline.rms_coverage, selected.rms_coverage, 3],
    ['STFT coverage', baseline.beat_v2?.coverage_stft, selected.beat_v2?.coverage_stft, 3],
    ['伴奏/人声 RMS dB', baseline.beat_v2?.acc_to_vocal_rms_db, selected.beat_v2?.acc_to_vocal_rms_db, 1],
    ['伴奏峰值', baseline.beat_v2?.stereo_peak, selected.beat_v2?.stereo_peak, 3],
    ['削波占比', baseline.beat_v2?.stereo_clipping_fraction, selected.beat_v2?.stereo_clipping_fraction, 5],
  ];
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
