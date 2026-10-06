const root = document.querySelector('#lyrics-formal');
const controls = root.querySelector('#formal-v2-controls');
const stages = [0, 1, 25, 50, 100, 270];

function showArm(arm, buttons) {
  for (const button of buttons) button.setAttribute('aria-pressed', String(button.arm === arm));
  root.querySelector('#formal-v2-status').textContent = '已提交 · 待两步 smoke';
  root.querySelector('#formal-v2-description').textContent =
    `${arm.model} · ${arm.reward} · LR ${arm.lr} · ${arm.host}。调度器作业 ${arm.smoke}（smoke）成功后，${arm.train} 才会启动 270 步训练。${arm.valid0 ? `固定验证 0/270 步已排队为 ${arm.valid0}/${arm.valid270}，依赖训练完成。` : ''}当前没有经核验的正式验证分数或配对音频。`;
  root.querySelector('#formal-v2-reward').textContent = '等待真实训练 reward 日志';
  root.querySelector('#formal-v2-kl').textContent = '等待真实训练 KL 日志';
  const rows = stages.map((step) => {
    const row = document.createElement('tr');
    for (const value of [String(step), '待测', '—', '待测', '待试听']) {
      const cell = document.createElement('td');
      cell.textContent = value;
      row.append(cell);
    }
    return row;
  });
  root.querySelector('#formal-v2-validation').replaceChildren(...rows);
}

try {
  const response = await fetch('./formal-9to1-runs.json', {cache: 'no-store'});
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const data = await response.json();
  if (data.protocol !== 'cmi-pref-triple-source-text-lyrics-baseline-9to1-v2' ||
      data.train_conditions !== 270 || data.valid_conditions !== 30 || data.sealed_test_conditions !== 121) {
    throw new Error('Formal data protocol mismatch');
  }
  const buttons = data.arms.map((arm) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = `${arm.model} · ${arm.reward} · ${arm.lr}${arm.host.startsWith('Kaya') ? ' · Kaya' : ''}`;
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
