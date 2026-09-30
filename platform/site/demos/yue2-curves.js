const section = document.querySelector('#yue2-training');
const canvas = document.querySelector('#yue2-reward-chart');
const value = document.querySelector('#yue2-reward-value');
let runs;

function draw() {
  if (!runs || section.hidden) return;
  const arm = document.querySelector('[data-replay-model][aria-pressed="true"]')?.dataset.replayModel;
  section.hidden = !arm?.startsWith('yue2');
  if (section.hidden) return;
  const points = runs[arm]?.points;
  if (!points) return;
  const width = canvas.getBoundingClientRect().width;
  if (width < 20) return;
  const height = 154;
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const context = canvas.getContext('2d');
  context.scale(ratio, ratio);
  const margin = { left: 39, right: 12, top: 12, bottom: 24 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const scores = points.map((point) => point.reward);
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
    context.fillText(score.toFixed(3), margin.left - 5, position + 4);
  }
  context.textAlign = 'center';
  for (const step of [0, 50, 100]) context.fillText(String(step), x(step), height - 4);
  context.strokeStyle = '#08745d';
  context.lineWidth = 2.5;
  context.lineJoin = 'round';
  context.lineCap = 'round';
  context.beginPath();
  points.forEach((point, index) => index ? context.lineTo(x(point.step), y(point.reward)) :
    context.moveTo(x(point.step), y(point.reward)));
  context.stroke();
  value.textContent = `${points[0].reward.toFixed(3)} → ${points.at(-1).reward.toFixed(3)}`;
}

try {
  const response = await fetch('./yue2-training-curves.json');
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  ({ runs } = await response.json());
  new ResizeObserver(draw).observe(canvas.parentElement);
  document.querySelectorAll('[data-replay-model]').forEach((button) => button.addEventListener('click', () => {
    section.hidden = !button.dataset.replayModel.startsWith('yue2');
    requestAnimationFrame(draw);
  }));
  draw();
} catch (error) {
  value.textContent = `曲线不可用：${error.message}`;
}
