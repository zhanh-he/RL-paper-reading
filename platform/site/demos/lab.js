const $ = (selector) => document.querySelector(selector);
const fmt = (value, digits = 4) => Number(value).toFixed(digits);
const signed = (value, digits = 3) => `${value >= 0 ? '+' : ''}${fmt(value, digits)}`;
const armLabels = {
  baseline: 'Frozen checkpoint',
  onset: 'GRPO · onset only',
  frame: 'GRPO · frame only',
  offset: 'GRPO · offset only',
  combined: 'GRPO · combined',
  bce: 'Weighted BCE control',
};
const auditNames = {
  onset: ['Pitch + onset', 'No voice assignment'],
  track_onset: ['Voice + onset', 'SATB aware'],
  track_note: ['Voice + complete note', 'Onset and offset'],
  active_voice: ['Active voice', '0.5 s windows'],
  offset: ['Pitch + offset', 'No voice assignment'],
};
const auditLessons = {
  swap_soprano_alto: '普通 onset F1 仍为 1.0，因此会奖励声部错位的结果。主 reward 应采用 SATB track-aware note F1，并单独观察 per-voice recall。',
  drop_bass: '只靠 overall F1 时，整条 Bass 消失仍能得到较高分数。应加 per-voice recall 的低分惩罚或 minimum-voice coverage，但避免强制每个无声窗口都四声部齐唱。',
  shift_onset_100ms: '0.5 秒 active-voice 分数几乎不变，却可能错过整批 note 边界。需要 onset 容差匹配与 offset/持续时间准确度；100 ms 在当前 50 ms onset 容差之外。',
  fragment_sustained: 'active-voice 几乎看不见把一条长音切成碎音符。使用 one-to-one note matching，加 fragmentation penalty，防止重复触发刷 recall。',
  overfill_octave: 'active-voice 对额外八度完全失明。note precision 和超范围/音高重复诊断能限制过填；单用 coverage/richness 会奖励这类错误。',
};

function setView(view) {
  const chosen = ['overview', 'choral', 'lyrics', 'songeval', 'rewards', 'vocal'].includes(view) ? view : 'overview';
  for (const section of document.querySelectorAll('.lab-view')) section.hidden = section.id !== `view-${chosen}`;
  for (const tab of document.querySelectorAll('.lab-tab')) tab.setAttribute('aria-selected', String(tab.dataset.view === chosen));
  document.querySelector(`.lab-tab[data-view="${chosen}"]`).scrollIntoView({ block: 'nearest', inline: 'center' });
  if (location.hash !== `#${chosen}`) history.replaceState(null, '', `#${chosen}`);
}

function renderChoral(data) {
  const runs = data.runs;
  $('#headline-grpo').textContent = `+${fmt(runs.frame.delta_f1.frame, 3)} F1`;
  $('#headline-bce').textContent = `+${fmt(runs.bce.delta_f1.frame, 3)} F1`;
  $('#frame-before').textContent = fmt(data.baseline_f1.frame);
  $('#frame-after').textContent = fmt(runs.frame.f1.frame);
  $('#frame-ci').textContent = `[+${fmt(runs.frame.delta_ci95.frame[0])}, +${fmt(runs.frame.delta_ci95.frame[1])}]`;
  const keys = ['baseline', 'onset', 'frame', 'offset', 'combined', 'bce'];
  const getF1 = (key, head) => key === 'baseline' ? data.baseline_f1[head] : runs[key].f1[head];
  $('#choral-table').replaceChildren(...keys.map((key) => {
    const tr = document.createElement('tr');
    const values = [armLabels[key], ...['onset', 'frame', 'offset'].map((head) => fmt(getF1(key, head))), key === 'baseline' ? '—' : `+${fmt(runs[key].delta_f1.frame)}`];
    for (const value of values) { const cell = document.createElement('td'); cell.textContent = value; tr.append(cell); }
    return tr;
  }));
  function renderChart() {
    const head = $('#head-select').value;
    const max = Math.max(...keys.map((key) => getF1(key, head)));
    $('#choral-chart').replaceChildren(...keys.map((key) => {
      const row = document.createElement('div'); row.className = `bar-row ${key === 'baseline' ? 'baseline' : key === 'bce' ? 'supervised' : ''}`;
      const label = document.createElement('span'); label.className = 'bar-label'; label.textContent = armLabels[key]; label.title = armLabels[key];
      const track = document.createElement('div'); track.className = 'bar-track';
      const fill = document.createElement('span'); fill.className = 'bar-fill'; fill.style.width = `${(getF1(key, head) / max) * 100}%`; track.append(fill);
      const value = document.createElement('span'); value.className = 'bar-value'; value.textContent = fmt(getF1(key, head), 3);
      row.append(label, track, value); return row;
    }));
  }
  $('#head-select').addEventListener('change', renderChart);
  renderChart();
}

function renderRewards(data) {
  $('#headline-swap').textContent = `${fmt(data.controlled.swap_soprano_alto.track_note.mean_f1, 2)} track F1`;
  const metricOrder = ['onset', 'track_onset', 'track_note', 'active_voice', 'offset'];
  function renderVariant() {
    const key = $('#variant-select').value;
    const current = data.controlled[key];
    $('#audit-metrics').replaceChildren(...metricOrder.map((metric) => {
      const item = document.createElement('div'); item.className = 'audit-metric';
      const title = document.createElement('span'); title.textContent = auditNames[metric][0];
      const value = document.createElement('strong'); value.textContent = fmt(current[metric].mean_f1, 3);
      const sub = document.createElement('small'); sub.textContent = auditNames[metric][1];
      item.append(title, value, sub); return item;
    }));
    $('#audit-conclusion').textContent = auditLessons[key];
  }
  $('#variant-select').addEventListener('change', renderVariant);
  renderVariant();
  const roles = { onset: 'Pitch timing only', offset: 'Offset timing only', track_onset: 'Voice + timing', track_note: 'Full note matching', active_voice: 'Auxiliary coverage signal' };
  $('#real-audit-table').replaceChildren(...metricOrder.map((metric) => {
    const tr = document.createElement('tr');
    for (const value of [auditNames[metric][0], fmt(data.real_predictions[metric].mean_f1), roles[metric]]) { const cell = document.createElement('td'); cell.textContent = value; tr.append(cell); }
    return tr;
  }));
}

function appendCells(body, rows) {
  body.replaceChildren(...rows.map((values) => {
    const tr = document.createElement('tr');
    for (const value of values) {
      const cell = document.createElement('td'); cell.textContent = value; tr.append(cell);
    }
    return tr;
  }));
}

function renderChoralNotes(data) {
  const rows = [
    ['Frame (16 ms)', 'frame_16ms', 'track_frame_16ms'],
    ['Onset (50 ms)', 'note_onset_50ms', 'track_note_onset_50ms'],
    ['Onset + offset (50 ms minimum)', 'note_onset_offset_50ms', 'track_note_onset_offset_50ms'],
  ];
  appendCells($('#choral-note-table'), rows.map(([label, pitch, track]) => [
    label, fmt(data.before[pitch].f1), fmt(data.before[track].f1), '相同',
  ]));
}

function pairRows(data) {
  return [['Before', data.before], ['After', data.after]].map(([name, entry]) => [
    name, fmt(entry.reward.mean), fmt(entry.signal.peak, 3),
    `${(entry.signal.near_full_scale_fraction * 100).toFixed(4)}%`,
  ]);
}

function renderLyrics(data, verification, muse) {
  appendCells($('#yue2-table'), pairRows(data));
  $('#yue2-token-change').textContent = `${verification.post_update_semantic_changed_positions}/${verification.semantic_tokens}`;
  $('#yue2-reward-change').textContent = `${fmt(data.before.reward.mean)} → ${fmt(data.after.reward.mean)}`;
  if (muse) {
    $('#muse-result').hidden = false;
    appendCells($('#muse-table'), pairRows(muse));
    $('#muse-interpretation').textContent = `同 seed 的留出片段，SongEval ${fmt(muse.before.reward.mean)} → ${fmt(muse.after.reward.mean)}；未训练重放波形${muse.untrained_replay_audio_identical ? '完全相同' : '不相同，不能归因于训练'}。单步、小样本结果不代表长期趋势。`;
  }
}

function renderSongEval(data) {
  const pairs = [
    ['Safe gain − normalized reference', 'safe_gain_minus_reference'],
    ['Full-scale clip − normalized reference', 'hard_clip_full_scale_minus_reference'],
    ['RMS-matched clip − safe gain', 'hard_clip_rms_matched_minus_safe_gain'],
    ['6 kHz bandlimit − reference', 'bandlimit_6khz_upsampled_minus_reference'],
  ];
  appendCells($('#songeval-table'), pairs.map(([label, key]) => {
    const score = data.comparisons[key].five_mean;
    return [label, signed(score.mean), `[${signed(score.paired_bootstrap_95pct[0])}, ${signed(score.paired_bootstrap_95pct[1])}]`, `${score.positive} / ${score.n}`];
  }));
}

for (const tab of document.querySelectorAll('.lab-tab')) tab.addEventListener('click', () => setView(tab.dataset.view));
for (const link of document.querySelectorAll('[data-open-view]')) link.addEventListener('click', (event) => { event.preventDefault(); setView(link.dataset.openView); window.scrollTo({ top: 0, behavior: 'smooth' }); });
window.addEventListener('hashchange', () => setView(location.hash.slice(1)));
setView(location.hash.slice(1) || 'overview');
if (window.lucide) window.lucide.createIcons();
try {
  const response = await fetch('./results.json');
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const data = await response.json();
  renderChoral(data.choral);
  renderChoralNotes(data.choral_note);
  $('#seed-range').textContent = [data.choral.runs.frame.f1.frame, ...data.choral_seed_repeats.map((row) => row.frame_f1)].map((score) => fmt(score)).join(' / ');
  renderRewards(data.reward_audit);
  renderLyrics(data.yue2, data.yue2_verification, data.muse);
  renderSongEval(data.songeval_audit);
} catch (error) {
  $('#headline-grpo').textContent = 'Data unavailable';
  $('#headline-bce').textContent = 'Data unavailable';
  $('#headline-swap').textContent = 'Data unavailable';
  console.error('Could not load experiment receipts', error);
}
