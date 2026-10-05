const root = document.getElementById('lyrics-formal');
const find = (selector) => root.querySelector(selector);
const number = (value, digits = 3) => Number.isFinite(value) ? value.toFixed(digits) : '—';
const checkpoints = [0, 1, 25, 50, 100];
let arms = {};
let selected = 'songeval_lr2e5';

function node(tag, value) {
  const element = document.createElement(tag);
  element.textContent = value;
  return element;
}

function chart(canvas, field, color) {
  const tooltip = node('div', '');
  tooltip.className = 'training-chart-tooltip';
  tooltip.hidden = true;
  canvas.parentElement.append(tooltip);
  let points = [];
  let verified = new Set();
  const height = 190;
  const margin = { left: 47, right: 13, top: 15, bottom: 27 };
  const transform = field === 'sampled_kl' ? (value) => Math.asinh(value / 0.05) : (value) => value;
  const smooth = (data, index) => {
    const window = data.slice(Math.max(0, index - 9), index + 1);
    return window.reduce((sum, point) => sum + point[field], 0) / window.length;
  };

  function draw() {
    const width = canvas.getBoundingClientRect().width;
    if (width < 80) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    const ctx = canvas.getContext('2d');
    ctx.scale(ratio, ratio);
    ctx.font = '11px system-ui, sans-serif';
    if (!points.length) {
      ctx.fillStyle = '#68726d';
      ctx.textAlign = 'center';
      ctx.fillText('等待已核验训练日志', width / 2, height / 2);
      return;
    }
    const values = points.map((point) => point[field]);
    const low = Math.min(...values);
    const high = Math.max(...values);
    const pad = Math.max((high - low) * 0.12, 0.015);
    const bottom = field === 'sampled_kl' ? 0 : low - pad;
    const top = field === 'sampled_kl'
      ? ([0.03, 0.1, 0.3, 1, 3, 10, 30, 100].find((tick) => tick >= high * 1.05) || high * 1.1)
      : high + pad;
    const x = (step) => margin.left + (width - margin.left - margin.right) * step / 100;
    const y = (value) => margin.top + (height - margin.top - margin.bottom)
      * (transform(top) - transform(value)) / Math.max(1e-9, transform(top) - transform(bottom));
    const ticks = field === 'sampled_kl'
      ? [0, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100].filter((tick) => tick <= top)
      : [bottom, (bottom + top) / 2, top];
    ctx.textAlign = 'right';
    ctx.fillStyle = '#68726d';
    let previousY = Infinity;
    for (const tick of ticks) {
      const tickY = y(tick);
      if (previousY - tickY < 18) continue;
      previousY = tickY;
      ctx.strokeStyle = '#dbe2dd';
      ctx.beginPath(); ctx.moveTo(margin.left, tickY); ctx.lineTo(width - margin.right, tickY); ctx.stroke();
      ctx.fillText(Math.abs(tick) < 1 ? tick.toFixed(tick === 0 ? 0 : 2) : tick.toFixed(1), margin.left - 6, tickY + 4);
    }
    ctx.textAlign = 'center';
    for (const step of [0, 25, 50, 100]) ctx.fillText(String(step), x(step), height - 5);
    for (const [key, alpha, lineWidth] of [[field, 0.2, 1], ['window', 1, 2.5]]) {
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.globalAlpha = alpha;
      ctx.lineWidth = lineWidth;
      points.forEach((point, index) => {
        if (index) ctx.lineTo(x(point.step), y(point[key]));
        else ctx.moveTo(x(point.step), y(point[key]));
      });
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
    for (const step of verified) {
      const point = points.find((item) => item.step === step);
      if (!point) continue;
      ctx.beginPath(); ctx.arc(x(step), y(point.window), 4, 0, Math.PI * 2);
      ctx.fillStyle = '#fff'; ctx.fill(); ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.stroke();
    }
  }

  canvas.addEventListener('pointermove', (event) => {
    if (!points.length) return;
    const bounds = canvas.getBoundingClientRect();
    const step = Math.max(0, Math.min(100, (event.clientX - bounds.left - margin.left)
      / (bounds.width - margin.left - margin.right) * 100));
    const point = points.reduce((best, item) => Math.abs(item.step - step) < Math.abs(best.step - step) ? item : best);
    tooltip.textContent = `${point.step} 步 · 10 步均值 ${number(point.window, 4)}\n本步原始值 ${number(point[field], 4)}`;
    tooltip.style.left = `${Math.max(80, Math.min(bounds.width - 80, event.clientX - bounds.left))}px`;
    tooltip.style.top = `${canvas.offsetTop + 8}px`;
    tooltip.hidden = false;
  });
  canvas.addEventListener('pointerleave', () => { tooltip.hidden = true; });
  new ResizeObserver(draw).observe(canvas.parentElement);
  return {
    set(raw, validated) {
      const finite = raw.filter((point) => Number.isInteger(point.step) && Number.isFinite(point[field]));
      points = finite.map((point, index) => ({ ...point, window: smooth(finite, index) }));
      verified = new Set(validated);
      tooltip.hidden = true;
      draw();
      return points;
    },
  };
}

const rewardChart = chart(find('#formal-reward-chart'), 'reward', '#08745d');
const klChart = chart(find('#formal-kl-chart'), 'sampled_kl', '#b36b24');

function render() {
  const arm = arms[selected];
  if (!arm) return;
  const validated = Object.keys(arm.validation).map(Number);
  const reward = rewardChart.set(arm.train_curve, validated);
  const kl = klChart.set(arm.train_curve, validated);
  find('#formal-reward-name').textContent = arm.reward_name;
  find('#formal-arm-status').textContent = arm.status === 'verified-complete'
    ? '完整核验' : arm.status === 'failed' ? '失败 · 见日志' : '已提交 · 待核验';
  find('#formal-arm-status').className = `state ${arm.status === 'verified-complete' ? 'measured' : 'pending'}`;
  find('#formal-reward-value').textContent = reward.length ? `${number(reward[0].window)} → ${number(reward.at(-1).window)}` : '待训练日志';
  find('#formal-kl-value').textContent = kl.length ? `${number(kl[0].window)} → ${number(kl.at(-1).window)}` : '待训练日志';
  find('#formal-training-extent').textContent = reward.length
    ? `1–${reward.at(-1).step} 步 · 每步 2 条采样` : '每步 2 条采样 · 100 步目标';
  find('#formal-training-note').textContent = reward.length
    ? `细线为单步原始值，粗线为最多 10 步滑动均值。圆点仅标记完整核验的 59 条验证 checkpoint。右图 sampled KL 针对冻结底座，β=0.01，asinh 非线性纵轴；与固定验证奖励是不同口径。训练日志到第 ${reward.at(-1).step} 步。`
    : `已提交 ${arm.host} smoke ${arm.smoke_job}；正式作业 ${arm.train_job} 依赖 smoke 成功。训练曲线与验证分数都尚未核验。`;
}

find('#formal-arm-select').addEventListener('change', (event) => {
  selected = event.target.value;
  render();
});

try {
  const response = await fetch('./formal-runs.json', { cache: 'no-store' });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const data = await response.json();
  arms = data.arms;
  const body = find('#formal-arms');
  body.replaceChildren(...Object.values(arms).map((arm) => {
    const baseline = arm.validation['0']?.mean_reward;
    const final = arm.validation['100']?.mean_reward;
    const row = node('tr', '');
    row.append(node('td', arm.label), node('td', `${arm.train_conditions} / ${arm.valid_conditions}`));
    for (const step of checkpoints) {
      const value = arm.validation[String(step)]?.mean_reward;
      row.append(node('td', Number.isFinite(value) ? number(value) : '待核验'));
    }
    row.append(node('td', Number.isFinite(baseline) && Number.isFinite(final)
      ? `${final - baseline >= 0 ? '+' : ''}${number(final - baseline)}` : '—'));
    return row;
  }));
  render();
} catch (error) {
  find('#formal-arm-status').textContent = '数据不可用';
  console.error('Could not load formal GRPO receipts', error);
}
