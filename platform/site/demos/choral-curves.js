const definitions = document.querySelector('#view-choral .choral-reward-details');
const foldouts = document.querySelector('#choral-training-foldouts');
foldouts.prepend(definitions);

const arms = [
  ['onset', 'Onset', '#08745d'],
  ['onset_offset', 'Onset + offset', '#ae6628'],
  ['frame', 'Frame', '#315fa8'],
  ['coverage', 'Coverage', '#9b5275'],
  ['continuity', 'Continuity', '#657a2b'],
];
const metricLabel = (value) => Math.abs(value) < 0.001 ? value.toFixed(6) : value.toFixed(3);

function chart(canvas, points, key, color, compact = false, rawPoints = null) {
  let selectedStep = 0;
  const tooltip = rawPoints ? document.createElement('div') : null;
  if (tooltip) {
    tooltip.className = 'training-chart-tooltip';
    tooltip.hidden = true;
    canvas.parentElement.append(tooltip);
  }
  const margin = { left: 39, right: 12, top: 12, bottom: 24 };
  const draw = () => {
    const width = canvas.getBoundingClientRect().width;
    if (width < 20) return;
    const height = compact ? 112 : 154;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    const context = canvas.getContext('2d');
    context.scale(ratio, ratio);
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    const values = [...points, ...(rawPoints || [])].map((point) => point[key]).filter(Number.isFinite);
    const low = Math.min(...values);
    const high = Math.max(...values);
    const pad = Math.max((high - low) * 0.13, 0.004);
    const minimum = Math.max(0, low - pad);
    const maximum = high + pad;
    const x = (step) => margin.left + plotWidth * step / 1000;
    const y = (value) => margin.top + plotHeight * (maximum - value) / (maximum - minimum);
    context.font = '11px system-ui, sans-serif';
    context.fillStyle = '#68726d';
    context.strokeStyle = '#dbe2dd';
    context.lineWidth = 1;
    for (let index = 0; index < 3; index++) {
      const value = minimum + (maximum - minimum) * index / 2;
      const position = y(value);
      context.beginPath();
      context.moveTo(margin.left, position);
      context.lineTo(width - margin.right, position);
      context.stroke();
      context.textAlign = 'right';
      context.fillText(value.toFixed(compact ? 2 : 3), margin.left - 5, position + 4);
    }
    context.textAlign = 'center';
    for (const step of [0, 500, 1000]) context.fillText(String(step), x(step), height - 4);
    if (selectedStep) {
      context.beginPath();
      context.setLineDash([3, 4]);
      context.strokeStyle = '#63716a';
      context.moveTo(x(selectedStep), margin.top);
      context.lineTo(x(selectedStep), height - margin.bottom);
      context.stroke();
      context.setLineDash([]);
    }
    context.lineJoin = 'round';
    context.lineCap = 'round';
    const trace = (series, alpha, thickness) => {
      context.beginPath();
      context.strokeStyle = color;
      context.globalAlpha = alpha;
      context.lineWidth = thickness;
      let connected = false;
      for (const point of series) {
        if (!Number.isFinite(point[key])) { connected = false; continue; }
        if (connected) context.lineTo(x(point.step), y(point[key]));
        else context.moveTo(x(point.step), y(point[key]));
        connected = true;
      }
      context.stroke();
      context.globalAlpha = 1;
    };
    if (rawPoints) trace(rawPoints, 0.2, 1);
    trace(points, 1, compact ? 2 : 2.5);
    for (const step of rawPoints ? [1, 100, 300, 1000] : []) {
      const point = rawPoints[step - 1];
      context.beginPath();
      context.arc(x(step), y(point[key]), step === selectedStep ? 5 : 3.5, 0, Math.PI * 2);
      context.fillStyle = '#fff';
      context.fill();
      context.strokeStyle = color;
      context.lineWidth = 2;
      context.stroke();
    }
  };
  if (tooltip) {
    canvas.addEventListener('pointermove', (event) => {
      const bounds = canvas.getBoundingClientRect();
      const plotWidth = bounds.width - margin.left - margin.right;
      if (plotWidth <= 0) return;
      const step = Math.max(1, Math.min(1000,
        Math.round((event.clientX - bounds.left - margin.left) / plotWidth * 1000)));
      const point = rawPoints[step - 1];
      const window = points[Math.floor((step - 1) / 25) + 1];
      tooltip.textContent = `第 ${step} 步 · 原始 Reward ${point.reward.toFixed(4)} · KL ${point.kl.toFixed(4)}\n25 步窗口均值 ${window?.[key]?.toFixed(4) ?? '—'}`;
      tooltip.style.left = `${Math.max(80, Math.min(bounds.width - 80, event.clientX - bounds.left))}px`;
      tooltip.style.top = `${canvas.offsetTop + 8}px`;
      tooltip.hidden = false;
    });
    canvas.addEventListener('pointerleave', () => { tooltip.hidden = true; });
  }
  new ResizeObserver(draw).observe(canvas.parentElement);
  draw();
  return { select(step) { selectedStep = step; draw(); } };
}

try {
  const response = await fetch('./choral-training-curves.json?v=20261006', { cache: 'no-store' });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const data = await response.json();
  const binned = data.runs.combined.points;
  const combined = [data.runs.combined.first_step, ...binned];
  const first = binned[0];
  const last = binned.at(-1);
  document.querySelector('#choral-combined-reward-value').textContent =
    `${first.reward.toFixed(3)} → ${last.reward.toFixed(3)}`;
  document.querySelector('#choral-combined-kl-value').textContent =
    `${first.kl.toFixed(3)} → ${last.kl.toFixed(3)}`;
  const rawPoints = data.runs.combined.raw_points;
  const rewardChart = chart(document.querySelector('#choral-combined-reward-chart'), combined,
    'reward', '#08745d', false, rawPoints);
  const klChart = chart(document.querySelector('#choral-combined-kl-chart'), combined,
    'kl', '#b36b24', false, rawPoints);
  const updateRail = () => {
    for (const button of document.querySelectorAll('#choral-event-rail button')) {
      const point = rawPoints[Number(button.dataset.step) - 1];
      button.querySelector('.state').textContent = `Reward ${metricLabel(point.reward)} · KL ${metricLabel(point.kl)}`;
    }
  };
  updateRail();
  document.querySelector('#view-choral .choral-training-note').textContent =
    '训练 rollout：细线为逐步原值，粗线为 25 步窗口均值；悬停任意位置显示对应 step 的 Reward、KL 原值。下方步骤同步标出虚线，均非未见歌曲 F1。两图纵轴独立缩放。';
  const selectStep = (step) => { rewardChart.select(step); klChart.select(step); };
  document.querySelector('#view-choral').addEventListener('choral:stepchange', (event) => {
    selectStep(event.detail.step);
    updateRail();
  });
  selectStep(Number(document.querySelector('#choral-event-rail button[aria-pressed="true"]')?.dataset.step) || 300);

  const list = document.querySelector('#choral-small-plots');
  for (const [arm, name, color] of arms) {
    const run = data.runs[arm];
    const figure = document.createElement('figure');
    const caption = document.createElement('figcaption');
    const label = document.createElement('span');
    label.textContent = name;
    const summary = document.createElement('small');
    summary.textContent = `${run.updated_steps}/1000 有效更新`;
    const canvas = document.createElement('canvas');
    canvas.setAttribute('role', 'img');
    canvas.setAttribute('aria-label', `${name} 单项训练 reward 随步数变化`);
    caption.append(label, summary);
    figure.append(caption, canvas);
    list.append(figure);
    chart(canvas, run.points, 'reward', color, true);
  }
} catch (error) {
  document.querySelector('.choral-training-note').textContent = `训练曲线暂不可用：${error.message}`;
}
