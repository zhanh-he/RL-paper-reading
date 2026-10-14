const root = document.querySelector('#lyrics-objective');
let dataset = 'wsb', origin = 'local', official, measured, priority;
const $ = (name) => root.querySelector(`[data-objective="${name}"]`);
const tr = (values, header = false) => {
  const row = document.createElement('tr');
  values.forEach((value) => {
    const c = document.createElement(header ? 'th' : 'td');
    if (typeof value === 'object') { c.textContent = value.text; c.title = value.reason || ''; c.dataset.status = value.status || ''; }
    else c.textContent = value;
    row.append(c);
  });
  return row;
};
const format = (key, value) => key === 'per' ? `${(100 * value).toFixed(2)}%` : value.toFixed(4);

function render() {
  root.querySelectorAll('[data-objective-dataset]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.objectiveDataset === dataset)));
  root.querySelectorAll('[data-objective-origin]').forEach((b) => {
    b.disabled = dataset === 'cmi' && b.dataset.objectiveOrigin === 'official';
    b.setAttribute('aria-pressed', String(b.dataset.objectiveOrigin === origin));
  });
  const published = origin === 'official';
  root.dataset.origin = origin;
  const rows = published ? official.rows : measured.datasets[dataset].rows;
  $('title').textContent = `${dataset === 'wsb' ? 'WildSongBench · 192 prompts' : 'CMI-Pref · 121 official test conditions'} · ${published ? '官方发表，非本站运行' : '本站冻结 checkpoint 实测'}`;
  $('head').replaceChildren(tr(['Model / experiment', published ? 'N' : 'Step · 生成 / 评分覆盖', ...official.metrics.map((m) => `${m.label} ${m.lowerIsBetter ? '↓' : '↑'}`)], true));
  $('body').replaceChildren(...rows.map((r) => tr([r.system,
    published ? r.n : `${r.status === 'waiting-for-training' ? '目标 ' : ''}${r.step} · ${r.generated_candidates}/${r.expected_candidates} 候选 · ${r.scored_conditions}/${measured.datasets[dataset].conditions} 条件`,
    ...official.metrics.map((m) => {
      if (published) return { text: format(m.key, r[m.key]), status: 'published', reason: 'YuE2 作者发表值，不是本站 baseline / GRPO 测量。' };
      const cell = r.metrics[m.key];
      let text = '待评估';
      if (cell.status === 'not-applicable') text = 'n/a';
      else if (cell.status === 'deferred') text = '暂缓';
      else if (cell.status === 'complete') text = format(m.key, cell.value);
      else if (cell.status === 'resource-pending') text = '待 80 GB';
      else if (cell.scored > 0) text = `待完成 (${cell.scored}/${cell.expected})`;
      else if (r.status === 'waiting-for-training') text = '待 checkpoint';
      else if (r.status === 'training-complete-not-evaluated') text = '训练完成 · 评估未提交';
      else if (r.status === 'training-resume-pending') text = '待续跑完成';
      else if (r.status === 'not-submitted') text = '未提交';
      return { text, status: cell.status, reason: cell.reason };
    })])));
  $('scope').textContent = published
    ? `来源：YuE2 / WSB 2026-09-12 官方快照，17 个设置、九列指标。同样 192 首，但不同模型的候选选择和算力不同，不能解释为等算力或本站 GRPO 的结果。`
    : dataset === 'wsb'
      ? '本站完整目标为 192 条 × 2 候选，每候选 4 次 ASR，按官方 PER / edits / phoneme length / candidate index 规则选择；其余指标共用同一首。YuE2 使用 legacy benchmark VAE、完整 symbolic planning 和 9000 semantic-token 上限，保留截断。尚未完整覆盖的均分不发布。'
      : '121 个官方 test 条件（125 偏好票），不使用偏好音频当唯一 ground truth。每条件两个固定种子，质量/对齐指标报告全部 242 候选的均值，不按质量挑最好一首。全体多语言 PER 与缺少冻结属性权重的 Q3O 标 n/a；不把英文 G2P 套在越南语、爪哇语等歌词上。';
  const productionJobs = measured.execution?.kaya.cmi_quality_jobs || [];
  const fullCampaign = measured.execution?.kaya.full_cmi_campaign;
  const submitted = measured.execution
    ? ` · Gadi wa66 / iv96：13 个 WSB 全量分片/续评任务${measured.execution.gadi.queue_snapshot ? '回收时全部排队' : '已提交'} · Kaya：${fullCampaign?.recovered_chunks ?? 0} 个新增批次 / ${fullCampaign?.recovered_candidates ?? 0} 候选评分已回收，另有 ${productionJobs.filter((j) => j.recovered).length} 个前缀批次；跨机核验不重复计入总分${measured.execution.q3o_deferred ? ' · Q3O 暂缓，不启动' : ''}`
    : '';
  const generated = measured.datasets.cmi.rows.filter((r) => r.id.startsWith('yue2-')).reduce((n, r) => n + r.generated_candidates, 0);
  const museRecords = measured.execution?.muse_training || [];
  const museDone = museRecords.filter((r) => r.verified_train_steps === 270).length;
  $('updated').textContent = `回收时间 ${new Date(measured.updated_at).toLocaleString('zh-CN', {timeZone: 'Australia/Perth'})} AWST · CMI 全量生成 ${generated}/1694 (${(100*generated/1694).toFixed(1)}%) · 5090 ${measured.queue_running_at_collection ? '运行中' : '回收时未运行'}${submitted} · 提交不代表完成；Muse ${museDone}/6 臂训练完成${museRecords.some((r) => r.resume) ? '、从 180 续跑已提交' : ''}，独立评估尚未提交`;
  if (priority?.other_work?.paused_on) $('updated').textContent = `2026-10-14 用户要求优先 SP / AI Rewards：此处旧评估队列和旧 Muse 续跑已暂停，保留已完成结果。以下数字为上次回收快照，不是当前运行状态：CMI 生成 ${generated}/1694；Muse ${museDone}/6 臂训练完成。提交或局部覆盖不等于全量完成。`;
  const diagnostics = measured.datasets[dataset].rows.flatMap((r) => (r.diagnostic_rows || []).map((sample) => ({ system: r.system, ...sample })));
  $('diagnostics').hidden = published || !diagnostics.length;
  $('diagnostic-head').replaceChildren(tr(['Experiment', 'Candidate · duration', ...official.metrics.map((m) => `${m.label} ${m.lowerIsBetter ? '↓' : '↑'}`)], true));
  $('diagnostic-body').replaceChildren(...diagnostics.map((r) => tr([r.system, `${r.id} · ${r.seconds.toFixed(1)} s`, ...official.metrics.map((m) => Number.isFinite(r.metrics[m.key]) ? format(m.key, r.metrics[m.key]) : dataset === 'cmi' && ['per', 'control_overall'].includes(m.key) ? 'n/a' : '待评估')])));
}

try {
  const load = async (file) => { const response = await fetch(file, { cache: 'no-store' }); if (!response.ok) throw new Error(`HTTP ${response.status}`); return response.json(); };
  official = await load('./data/wsb-official-results.json');
  measured = await load('./data/lyrics-objective-results.json');
  priority = await load('./data/reward-regime-protocol.json');
  if (official.metrics.length !== 9 || official.rows.length !== 17 || measured.schema !== 1) throw new Error('Objective evaluation schema mismatch');
  root.querySelectorAll('[data-objective-dataset]').forEach((b) => b.addEventListener('click', () => {
    dataset = b.dataset.objectiveDataset; if (dataset === 'cmi') origin = 'local'; render();
  }));
  root.querySelectorAll('[data-objective-origin]').forEach((b) => b.addEventListener('click', () => { origin = b.dataset.objectiveOrigin; render(); }));
  render();
} catch (error) { $('scope').textContent = '客观评估清单加载失败；未显示替代分数。'; console.error(error); }
