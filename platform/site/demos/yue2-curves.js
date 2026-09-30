import { createMilestones } from './chart-milestones.js';

const section = document.querySelector('#yue2-training');
const canvas = document.querySelector('#yue2-reward-chart');
const milestones = createMilestones(canvas, 100);
const value = document.querySelector('#yue2-reward-value');
const details = document.querySelector('#yue2-dimensions-details');
const dimensionList = document.querySelector('#yue2-dimension-plots');
const dimensions = [
  ['Coherence', '#08745d'],
  ['Musicality', '#ae6628'],
  ['Memorability', '#315fa8'],
  ['Clarity', '#9b5275'],
  ['Naturalness', '#657a2b'],
];
let runs;

function drawChart(canvas, points, key, color, compact = false, marks = null) {
  const width = canvas.getBoundingClientRect().width;
  if (width < 20) return;
  const height = compact ? 112 : 154;
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const context = canvas.getContext('2d');
  context.scale(ratio, ratio);
  const margin = { left: 39, right: 12, top: 12, bottom: 24 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const scores = points.map((point) => point[key]);
  const low = Math.min(...scores);
  const high = Math.max(...scores);
  const pad = Math.max((high - low) * 0.13, 0.004);
  const minimum = Math.max(0, low - pad);
  const maximum = high + pad;
  const x = (step) => margin.left + plotWidth * step / 100;
  const y = (score) => margin.top + plotHeight * (maximum - score) / (maximum - minimum);
  context.font = '11px system-ui, sans-serif';
  context.fillStyle = '#68726d';
  context.strokeStyle = '#dbe2dd';
  context.lineWidth = 1;
  for (let index = 0; index < 3; index++) {
    const score = minimum + (maximum - minimum) * index / 2;
    const position = y(score);
    context.beginPath();
    context.moveTo(margin.left, position);
    context.lineTo(width - margin.right, position);
    context.stroke();
    context.textAlign = 'right';
    context.fillText(score.toFixed(compact ? 2 : 3), margin.left - 5, position + 4);
  }
  context.textAlign = 'center';
  for (const step of [0, 50, 100]) context.fillText(String(step), x(step), height - 4);
  context.strokeStyle = color;
  context.lineWidth = compact ? 2 : 2.5;
  context.lineJoin = 'round';
  context.lineCap = 'round';
  context.beginPath();
  points.forEach((point, index) => index ? context.lineTo(x(point.step), y(point[key])) :
    context.moveTo(x(point.step), y(point[key])));
  context.stroke();
  if (compact) {
    context.fillStyle = color;
    for (const point of points) {
      context.beginPath();
      context.arc(x(point.step), y(point[key]), 2.5, 0, Math.PI * 2);
      context.fill();
    }
  }
  marks?.draw(context, width, height - margin.bottom, color);
}

function draw() {
  if (!runs || section.hidden) return;
  const arm = document.querySelector('[data-replay-model][aria-pressed="true"]')?.dataset.replayModel;
  const run = runs[arm];
  if (!run) return;
  const points = run.points;
  milestones.set(run.heldout.filter((point) => point.step > 0).map((point) => {
    const training = points.find((candidate) => candidate.step === point.step);
    return { step: point.step, label: `第 ${point.step} 步 · 留出 SongEval 均分 ${point.mean.toFixed(4)}${training ? `\n训练 rollout 5 步窗口均值 ${training.reward.toFixed(4)}` : '\n训练 rollout reward 未记录'}` };
  }));
  drawChart(canvas, points, 'reward', '#08745d', false, milestones);
  value.textContent = `${points[0].reward.toFixed(3)} → ${points.at(-1).reward.toFixed(3)}`;
  if (!details.open) return;
  for (const [key, color] of dimensions) {
    const plot = dimensionList.querySelector(`[data-dimension="${key}"]`);
    const before = run.heldout[0][key];
    const after = run.heldout.at(-1)[key];
    plot.querySelector('small').textContent = `${before.toFixed(3)} → ${after.toFixed(3)} (${after >= before ? '+' : ''}${(after - before).toFixed(3)})`;
    drawChart(plot.querySelector('canvas'), run.heldout, key, color, true);
  }
}

try {
  const response = await fetch('./yue2-training-curves.json');
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  ({ runs } = await response.json());
  for (const [key, color] of dimensions) {
    const plot = document.createElement('figure');
    plot.dataset.dimension = key;
    const caption = document.createElement('figcaption');
    const label = document.createElement('span');
    label.textContent = key;
    label.style.color = color;
    const summary = document.createElement('small');
    const chart = document.createElement('canvas');
    chart.setAttribute('role', 'img');
    chart.setAttribute('aria-label', `${key} 三首固定留出歌曲均分随 YuE2 检查点变化`);
    caption.append(label, summary);
    plot.append(caption, chart);
    dimensionList.append(plot);
  }
  new ResizeObserver(draw).observe(canvas.parentElement);
  details.addEventListener('toggle', draw);
  document.querySelectorAll('[data-replay-model]').forEach((button) => button.addEventListener('click', () => {
    section.hidden = !button.dataset.replayModel.startsWith('yue2');
    requestAnimationFrame(draw);
  }));
  draw();
} catch (error) {
  value.textContent = `曲线不可用：${error.message}`;
}
