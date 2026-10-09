const root = document.querySelector('#lyrics-formal');
const controls = root.querySelector('#formal-v2-controls');
const stages = [0, 1, 25, 50, 100, 270];

function showArm(arm, buttons) {
  for (const button of buttons) button.setAttribute('aria-pressed', String(button.arm === arm));
  const complete = arm.status === 'verified-complete';
  const status = root.querySelector('#formal-v2-status');
  status.textContent = complete ? '270 步 · collector 核验' : '快照时排队 · 无结果';
  status.className = `state ${complete ? 'measured' : 'pending'}`;
  const baseline = arm.validation['0'];
  const final = arm.validation['270'];
  const delta = Number.isFinite(baseline) && Number.isFinite(final) ? final - baseline : null;
  root.querySelector('#formal-v2-description').textContent = complete
    ? `${arm.model} · ${arm.reward} · LR ${arm.lr} · ${arm.host} · 270 步，作业 ${arm.train}。30 条固定验证条件：${baseline.toFixed(4)} → ${final.toFixed(4)}（Δ ${delta >= 0 ? '+' : ''}${delta.toFixed(4)}）。独立 collector 核验摘要见下方运行记录；官方 test 仍封存。`
    : `${arm.model} · ${arm.reward} · LR ${arm.lr} · ${arm.host}。${arm.train ? `v2 作业 ${arm.train} ` : '该臂'}在 2026-10-08 记录时排队；当前仓库无完成核验。实时队列状态未知。`;
  root.querySelector('#formal-v2-reward').textContent = '逐步训练日志未入库，暂无可信曲线';
  root.querySelector('#formal-v2-kl').textContent = '逐步 KL 日志未入库，暂无可信曲线';
  root.querySelector('#formal-v2-validation').replaceChildren(...stages.map((step) => {
    const value = arm.validation[String(step)];
    const row = document.createElement('tr');
    const cells = [String(step), Number.isFinite(value) ? value.toFixed(4) : '未公开',
      step === 270 && Number.isFinite(delta) ? `${delta >= 0 ? '+' : ''}${delta.toFixed(4)}` : '—',
      '未入库', '本轮无公开音频'];
    for (const content of cells) {
      const cell = document.createElement('td');
      cell.textContent = content;
      row.append(cell);
    }
    return row;
  }));
}

try {
  const response = await fetch('./formal-9to1-runs.json', { cache: 'no-store' });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const data = await response.json();
  if (data.protocol !== 'cmi-pref-triple-source-text-lyrics-baseline-9to1-v2' ||
      data.train_conditions !== 270 || data.valid_conditions !== 30 || data.sealed_test_conditions !== 121) {
    throw new Error('Formal data protocol mismatch');
  }
  const buttons = data.arms.map((arm) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = `${arm.model} · ${arm.reward} · ${arm.lr}${arm.host.startsWith('Kaya') ? ' · Kaya' : ''}${arm.status === 'verified-complete' ? ' ✓' : ' · 排队快照'}`;
    button.arm = arm;
    button.addEventListener('click', () => showArm(arm, buttons));
    controls.append(button);
    return button;
  });
  showArm(data.arms[0], buttons);
} catch (error) {
  root.querySelector('#formal-v2-status').textContent = '实验清单不可用';
  console.error('Could not load formal 9:1 experiment registry', error);
}
