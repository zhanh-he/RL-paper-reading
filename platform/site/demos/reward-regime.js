const root = document.querySelector('#view-reward-regimes');
const $ = (id) => root.querySelector(`[data-regime="${id}"]`);
const row = (values, heading = false) => {
  const tr = document.createElement('tr');
  for (const value of values) {
    const cell = document.createElement(heading ? 'th' : 'td');
    if (value instanceof Node) cell.append(value); else cell.textContent = value;
    tr.append(cell);
  }
  return tr;
};
let protocol, selected = 'yue2';
function render() {
  for (const b of $('models').querySelectorAll('button')) b.setAttribute('aria-pressed', String(b.dataset.model === selected));
  const model = protocol.models.find((m) => m.id === selected);
  $('model-note').textContent = `${model.name} · ${model.role} · ${model.status} · MOS 未测。${model.note}`;
  $('arms').replaceChildren(...protocol.rewards.map((r) => row([
    `${model.name} · ${r.name}`, r.status, '未授权 / 未启动', '—', '—', '—',
  ])));
}
try {
  const response = await fetch('./data/reward-regime-protocol.json', { cache: 'no-store' });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  protocol = await response.json();
  if (protocol.training_authorized || protocol.runs.length) throw new Error('Proposal unexpectedly contains active runs');
  $('hypothesis').textContent = protocol.hypothesis;
  $('models').replaceChildren(...protocol.models.map((m) => {
    const b = document.createElement('button'); b.type = 'button'; b.textContent = m.name; b.dataset.model = m.id;
    b.addEventListener('click', () => { selected = m.id; render(); }); return b;
  }));
  $('rewards').replaceChildren(...protocol.rewards.map((r) => {
    const link = document.createElement('a'); link.href = r.source; link.textContent = r.name;
    return row([link, r.definition, r.meaning, `${r.status}。${r.note}`]);
  }));
  $('data').replaceChildren(...['train', 'valid', 'test', 'modalities'].map((key) => row([key, protocol.data[key]])));
  $('design').replaceChildren(...Object.entries(protocol.design).map(([key, value]) => row([key, Array.isArray(value) ? value.join(' / ') : value])));
  render();
} catch (error) { $('model-note').textContent = '协议加载失败，未显示替代结果。'; console.error(error); }
