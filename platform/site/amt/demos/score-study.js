const $ = selector => document.querySelector(selector);
const element = (tag, value) => { const node = document.createElement(tag); node.textContent = value; return node; };
function source(label, href) { const node = element('a', label); node.href = href; node.target = '_blank'; node.rel = 'noreferrer'; return node; }
function cell(value) { const node = document.createElement('td'); node.append(typeof value === 'string' ? document.createTextNode(value) : value); return node; }
try {
  const response = await fetch('./data/score-study.json', { cache: 'no-store' });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const data = await response.json();
  $('#score-study-date').textContent = `核验 ${data.updated_at}`;
  if (data.results) {
    const {sheet,duet,sft,grpo} = data.results;
    const rows = [
      ['SheetSage2 · frozen', '10 recordings', 'Beat / downbeat F1 ↑', `${(sheet.beat_f1*100).toFixed(2)}% / ${(sheet.downbeat_f1*100).toFixed(2)}%`, '—', '真实 GTZAN 小子集；组件评测，无谱面真值'],
      ['DuET · duration-head SFT', '8 full works', 'MUSTER Mean ER ↓', `${duet.baseline.MeanER.toFixed(3)}%`, `${duet.sft50.MeanER.toFixed(3)}%`, `Δ ${duet.delta_mean_er_pp >= 0 ? '+' : ''}${duet.delta_mean_er_pp.toFixed(3)} pp；单 seed 描述性结果`],
      ['DuET · duration-head GRPO', '8 full works', 'MUSTER Mean ER ↓', `${duet.baseline.MeanER.toFixed(3)}%`, `${duet.grpo50.MeanER.toFixed(3)}%`, `${grpo.rollouts} rollout batches / ${grpo.optimizer_steps} updates；Δ +${duet.grpo_delta_mean_er_pp.toFixed(3)} pp；非整序列 GRPO`],
      ['Duration token · valid', `${sft.valid_works} works / ${sft.before.validation.n_notes} notes`, 'Exact duration accuracy ↑', `${(sft.before.validation.accuracy*100).toFixed(2)}%`, `${(sft.after.validation.accuracy*100).toFixed(2)}%`, `${sft.train_works} train works / ${sft.before.train.n_notes} labels；${sft.optimizer_steps} 次 head-only 更新`],
    ];
    $('#score-results-body').replaceChildren(...rows.map(values => { const row = document.createElement('tr'); row.append(...values.map(cell)); return row; }));
    const signed = value => `${value >= 0 ? '+' : ''}${value.toFixed(3)}`;
    $('#score-paired-body').replaceChildren(...duet.samples.map(item => { const row = document.createElement('tr'); row.append(...[item.work,`${item.before.toFixed(3)}%`,`${item.after.toFixed(3)}% (${signed(item.delta)})`,`${item.grpo.toFixed(3)}% (${signed(item.grpo_delta)})`].map(cell)); return row; }));
    $('#score-gtzan-body').replaceChildren(...sheet.samples.map(item => { const row = document.createElement('tr'); row.append(...[item.id,`${(item.beat_f1*100).toFixed(2)}%`,`${(item.downbeat_f1*100).toFixed(2)}%`].map(cell)); return row; }));
    $('#score-head-diagnostics').replaceChildren(...['train','validation'].map(split => {
      const row = document.createElement('tr');
      row.append(...[split, String(sft.before[split].n_notes), sft.before[split].cross_entropy.toFixed(3),sft.after[split].cross_entropy.toFixed(3),grpo.after[split].cross_entropy.toFixed(3), `${(sft.before[split].accuracy*100).toFixed(2)}% → ${(sft.after[split].accuracy*100).toFixed(2)}% / ${(grpo.after[split].accuracy*100).toFixed(2)}%`].map(cell));
      return row;
    }));
    $('#score-grpo-trace').replaceChildren(...grpo.trace.map(item => {
      const row = document.createElement('tr');
      row.append(...[String(item.step),item.reward.toFixed(4),item.kl.toFixed(6),String(item.variable_groups),item.optimizer_update ? 'Update' : 'Skipped · zero variance'].map(cell));
      return row;
    }));
  }
  $('#score-study-plan').replaceChildren(...data.experiments.map(item => {
    const row = document.createElement('tr');
    for (const value of [item.title, item.data, item.protocol, item.action, item.status]) row.append(cell(value));
    return row;
  }));
  let family = 'all';
  const names = { audio: 'Audio → Score', midi: 'MIDI → Score', cascade: 'Cascade', engraving: 'Engraving' };
  function renderModels() {
    const records = data.models.filter(model => family === 'all' || model.family === family);
    $('#score-model-count').textContent = `${records.length} / ${data.models.length} systems`;
    $('#score-model-body').replaceChildren(...records.map(model => {
      const row = document.createElement('tr');
      const identity = document.createElement('div'); identity.append(element('strong', model.name), element('small', names[model.family]));
      const refs = document.createElement('div'); refs.append(source('Paper ↗', model.paper));
      if (model.code !== model.paper) refs.append(document.createTextNode(' · '), source('Code / release ↗', model.code));
      identity.append(refs);
      for (const value of [identity, model.output, `${model.benchmark}\n${model.metrics}`, model.release, model.priority]) row.append(cell(value));
      return row;
    }));
  }
  for (const button of document.querySelectorAll('[data-score-family]')) button.addEventListener('click', () => {
    family = button.dataset.scoreFamily;
    for (const other of document.querySelectorAll('[data-score-family]')) other.setAttribute('aria-pressed', String(other === button));
    renderModels();
  });
  renderModels();
  for (const button of document.querySelectorAll('[data-study-open]')) button.addEventListener('click', () => {
    $(`[data-tab="${button.dataset.studyOpen}"]`).click();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });
  $('#score-data-body').replaceChildren(...data.datasets.map(record => {
    const row = document.createElement('tr');
    for (const value of [source(record.name, record.source), record.labels, record.split, record.limitation]) row.append(cell(value));
    return row;
  }));
  $('#score-rewards-body').replaceChildren(...data.rewards.map(record => {
    const row = document.createElement('tr');
    for (const value of [record.name, record.definition, record.guard, record.status]) row.append(cell(value));
    return row;
  }));
} catch (error) { $('#score-study-date').textContent = `实验计划加载失败：${error.message}`; }
