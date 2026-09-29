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
const schematicNotes = [
  ...[[72, 74, 76], [67, 69, 65], [60, 62, 59], [48, 52, 48]].flatMap((pitches, voice) => [
    { voice, pitch: pitches[0], start: 0, duration: 1.8 },
    { voice, pitch: pitches[1], start: 2, duration: 0.8 },
    { voice, pitch: pitches[2], start: 3, duration: 0.8 },
  ]),
];

function renderPianoRoll(variant) {
  const canvas = $('#reward-piano-roll');
  const ctx = canvas.getContext('2d');
  let changed = schematicNotes.map((note) => ({ ...note }));
  if (variant === 'swap_soprano_alto') changed = changed.map((note) => ({ ...note, voice: note.voice === 0 ? 1 : note.voice === 1 ? 0 : note.voice }));
  if (variant === 'drop_bass') changed = changed.filter((note) => note.voice !== 3);
  if (variant === 'shift_onset_100ms') changed = changed.map((note) => ({ ...note, start: note.start + 0.1 }));
  if (variant === 'fragment_sustained') {
    changed = changed.filter((note) => !(note.voice === 3 && note.start === 0));
    changed.push(...[0, 0.45, 0.9, 1.35].map((start) => ({ voice: 3, pitch: 48, start, duration: 0.4 })));
  }
  if (variant === 'overfill_octave') changed.push({ voice: 0, pitch: 88, start: 2, duration: 0.8 });
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#ffffff'; ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.font = '12px system-ui, sans-serif';
  ctx.fillStyle = '#176b57'; ctx.fillRect(154, 10, 17, 11); ctx.fillStyle = '#17211e'; ctx.fillText('Reference', 178, 20);
  ctx.fillStyle = '#b2473c'; ctx.fillRect(274, 10, 17, 11); ctx.fillStyle = '#17211e'; ctx.fillText('Controlled variant', 298, 20);
  const x0 = 154, scale = 220, top = 38, laneHeight = 80;
  for (let voice = 0; voice < 4; voice++) {
    const y = top + voice * laneHeight;
    ctx.fillStyle = voice % 2 ? '#f5f7f5' : '#fbfcfb'; ctx.fillRect(0, y, canvas.width, laneHeight);
    ctx.fillStyle = '#17211e'; ctx.font = 'bold 13px system-ui, sans-serif'; ctx.fillText(['Soprano', 'Alto', 'Tenor', 'Bass'][voice], 16, y + 34);
    ctx.fillStyle = '#64736c'; ctx.font = '11px system-ui, sans-serif'; ctx.fillText('ref', 114, y + 30); ctx.fillText('test', 114, y + 60);
    ctx.strokeStyle = '#d7ddd9'; ctx.beginPath(); ctx.moveTo(0, y + laneHeight); ctx.lineTo(canvas.width, y + laneHeight); ctx.stroke();
  }
  for (let second = 0; second <= 4; second++) {
    const x = x0 + second * scale;
    ctx.strokeStyle = '#ccd5cf'; ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top + 4 * laneHeight); ctx.stroke();
    ctx.fillStyle = '#5f6b66'; ctx.font = '11px system-ui, sans-serif'; ctx.fillText(`${second}s`, x + 3, canvas.height - 5);
  }
  function draw(notes, color, offset) {
    for (const note of notes) {
      const x = x0 + note.start * scale, y = top + note.voice * laneHeight + offset;
      const width = Math.max(6, note.duration * scale - 4);
      ctx.fillStyle = color; ctx.fillRect(x, y, width, 18);
      ctx.fillStyle = '#ffffff'; ctx.font = 'bold 11px system-ui, sans-serif';
      if (width > 26) ctx.fillText(String(note.pitch), x + 5, y + 13);
    }
  }
  draw(schematicNotes, '#176b57', 15);
  draw(changed, '#b2473c', 44);
}

function setView(view) {
  const chosen = ['overview', 'choral', 'lyrics', 'songeval', 'rewards', 'vocal'].includes(view) ? view : 'lyrics';
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
    renderPianoRoll(key);
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

function renderStageRail(selector, stages, onSelect = null, selectedStep = null) {
  $(selector).replaceChildren(...stages.map((stage) => {
    const selectable = onSelect && stage.status === 'measured' && stage.step > 0;
    const item = document.createElement(selectable ? 'button' : 'div');
    if (selectable) { item.type = 'button'; item.setAttribute('aria-pressed', String(stage.step === selectedStep)); item.addEventListener('click', () => onSelect(stage)); }
    item.className = `stage-item ${stage.status === 'measured' ? 'available' : ''}`;
    const title = document.createElement('strong');
    title.textContent = stage.step === 0 ? 'Baseline' : `${stage.step} ${stage.step === 1 ? 'step' : 'steps'}`;
    const state = document.createElement('span');
    state.className = `state ${stage.status === 'measured' ? 'measured' : stage.status === 'metrics_only' ? 'diagnostic' : 'pending'}`;
    state.textContent = stage.status === 'measured' ? '可试听' : stage.status === 'metrics_only' ? '仅指标' : '待运行';
    item.append(title, state);
    return item;
  }));
}

function createMediaPanel({ label, title, asset, details, listenRole }) {
  const panel = document.createElement('article'); panel.className = 'compare-panel';
  if (listenRole) panel.dataset.listenPanel = listenRole;
  const index = document.createElement('span'); index.className = 'audio-index'; index.textContent = label;
  const heading = document.createElement('h3'); heading.textContent = title;
  const wave = document.createElement('div'); wave.className = 'wave-visual';
  const waveImage = document.createElement('img'); waveImage.src = `./${asset.wave}`; waveImage.alt = `${title} waveform, full duration`; waveImage.loading = 'lazy';
  const playhead = document.createElement('span'); playhead.className = 'wave-playhead'; playhead.setAttribute('aria-hidden', 'true');
  wave.append(waveImage, playhead);
  const spectrum = document.createElement('figure'); spectrum.className = 'spectrum-visual';
  const spectrumImage = document.createElement('img'); spectrumImage.src = `./${asset.spectrum}`; spectrumImage.alt = `${title} spectrogram, 40 Hz to 16 kHz on a logarithmic frequency scale`; spectrumImage.loading = 'lazy';
  const caption = document.createElement('figcaption'); caption.textContent = '40 Hz – 16 kHz · same visual scale';
  spectrum.append(spectrumImage, caption);
  const audio = document.createElement('audio'); audio.controls = true; audio.preload = 'metadata'; audio.src = `./${asset.audio}`;
  const playingLabel = () => listenRole === 'baseline' ? 'A · Baseline' : document.querySelector('[data-listen="candidate"]').textContent;
  audio.addEventListener('timeupdate', () => {
    wave.style.setProperty('--progress', `${audio.duration ? 100 * audio.currentTime / audio.duration : 0}%`);
    if (listenRole && !audio.paused) $('#listen-status').textContent = `${playingLabel()} · ${Math.floor(audio.currentTime)}s`;
  });
  audio.addEventListener('play', () => {
    for (const other of document.querySelectorAll('audio')) if (other !== audio) other.pause();
    if (listenRole) {
      for (const button of document.querySelectorAll('[data-listen]')) button.setAttribute('aria-pressed', String(button.dataset.listen === listenRole));
      $('#listen-status').textContent = `${playingLabel()} · ${Math.floor(audio.currentTime)}s`;
    }
  });
  const note = document.createElement('small'); note.textContent = details;
  panel.append(index, heading, wave, spectrum, audio, note);
  return panel;
}

function renderLyrics(data) {
  const models = data.replays.lyrics;
  let currentModel = 'yue2';
  let currentStep = 1;
  function renderModel(model, step = 1) {
    currentModel = model;
    currentStep = step;
    const descriptor = models[model];
    const receipt = data[model];
    const baseline = descriptor.stages[0];
    const selected = descriptor.stages.find((stage) => stage.step === step && stage.status === 'measured');
    if (!selected) throw new Error(`No measured stage ${step} for ${model}`);
    $('#lyrics-style').textContent = descriptor.style;
    $('#lyrics-text').textContent = descriptor.lyrics;
    for (const button of document.querySelectorAll('[data-replay-model]')) button.setAttribute('aria-pressed', String(button.dataset.replayModel === model));
    renderStageRail('#lyrics-stage-rail', descriptor.stages, (stage) => renderModel(model, stage.step), step);
    const pending = descriptor.stages.filter((stage) => stage.status === 'pending').map((stage) => stage.step);
    $('#lyrics-stage-context').textContent = `${pending.length ? `${pending.join(' / ')} steps 待运行。` : ''}欠拟合、改善或 reward hacking 必须由留出音频与指标共同判断，不能按步数预设。`;
    document.querySelector('[data-listen="candidate"]').textContent = `B · ${step} ${step === 1 ? 'step' : 'steps'}`;
    $('#lyrics-metric-head').textContent = `${step} ${step === 1 ? 'step' : 'steps'}`;
    $('#lyrics-compare').replaceChildren(
      createMediaPanel({ label: 'A / BASELINE', title: `${descriptor.name} · 0 updates`, asset: baseline, details: '48 kHz · held-out seed 5101', listenRole: 'baseline' }),
      createMediaPanel({ label: 'B / GRPO', title: `${descriptor.name} · ${step} ${step === 1 ? 'update' : 'updates'}`, asset: selected, details: '同 prompt、同 seed · 独立留出样本', listenRole: 'candidate' }),
    );
    const before = baseline.metrics, after = selected.metrics;
    const metrics = [
      ...['Coherence', 'Musicality', 'Memorability', 'Clarity', 'Naturalness', 'mean'].map((key) => [`SongEval · ${key}`, before.reward[key], after.reward[key], 4]),
      ['Audio · peak', before.signal.peak, after.signal.peak, 3],
      ['Audio · RMS', before.signal.rms, after.signal.rms, 3],
      ['Audio · ≥0.999 samples (%)', 100 * before.signal.near_full_scale_fraction, 100 * after.signal.near_full_scale_fraction, 4],
    ];
    appendCells($('#lyrics-metric-table'), metrics.map(([name, a, b, digits]) => [name, fmt(a, digits), fmt(b, digits), signed(b - a, digits)]));
    if (step === 1) {
      const changed = model === 'yue2' ? `${data.yue2_verification.post_update_semantic_changed_positions}/${data.yue2_verification.semantic_tokens} semantic tokens` : `${receipt.heldout_generated_token_difference}/${receipt.max_new_tokens} generated tokens`;
      $('#lyrics-interpretation').textContent = `未训练重放波形完全相同；更新后 ${changed} 改变。本次留出片段 SongEval ${fmt(before.reward.mean)} → ${fmt(after.reward.mean)}，两版均无满幅削波。只有 2 条训练 rollout、1 次优化与 1 条留出样本；不足以判断长程优化、听感改善或 reward hacking。`;
    } else {
      $('#lyrics-interpretation').textContent = `${step} 步留出片段 SongEval ${fmt(before.reward.mean)} → ${fmt(after.reward.mean)}。该样本的主观偏好与其他留出样本也必须核对，不能只凭 reward 认定改善。`;
    }
    $('#listen-status').textContent = '选择 A 或 B 开始播放';
    for (const button of document.querySelectorAll('[data-listen]')) button.setAttribute('aria-pressed', 'false');
    const form = $('#listening-review');
    form.reset();
    try {
      const review = JSON.parse(localStorage.getItem(`music-review-${model}-step${step}`) || 'null');
      if (review) { form.elements.preference.value = review.preference; form.elements.note.value = review.note; }
      $('#review-status').textContent = review ? '本机已保存' : '';
    } catch { $('#review-status').textContent = ''; }
  }
  for (const button of document.querySelectorAll('[data-replay-model]')) button.addEventListener('click', () => renderModel(button.dataset.replayModel));
  for (const button of document.querySelectorAll('[data-listen]')) button.addEventListener('click', () => {
    const target = $(`[data-listen-panel="${button.dataset.listen}"] audio`);
    const other = $(`[data-listen-panel="${button.dataset.listen === 'baseline' ? 'candidate' : 'baseline'}"] audio`);
    const time = Number.isFinite(other.currentTime) ? other.currentTime : 0;
    other.pause();
    const start = () => { target.currentTime = Math.min(time, Math.max(0, target.duration - 0.1)); target.play().catch(() => { $('#listen-status').textContent = '请直接点击音频播放器'; }); };
    if (target.readyState >= 1) start(); else target.addEventListener('loadedmetadata', start, { once: true });
  });
  $('#listening-review').addEventListener('submit', (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    try {
      localStorage.setItem(`music-review-${currentModel}-step${currentStep}`, JSON.stringify({ preference: form.elements.preference.value, note: form.elements.note.value.trim() }));
      $('#review-status').textContent = '已保存于本机浏览器';
    } catch { $('#review-status').textContent = '浏览器未允许本机保存'; }
  });
  renderModel(currentModel);
}

function renderVocal(replay) {
  renderStageRail('#vocal-stage-rail', replay.stages);
  $('#vocal-compare').replaceChildren(
    createMediaPanel({ label: 'INPUT', title: 'Synthetic vocal guide', asset: replay.input, details: 'Original synthetic asset · 16 s · mono' }),
    createMediaPanel({ label: 'BASELINE / NO GRPO', title: 'ACE-Step 1.5 completion', asset: replay.stages[0], details: 'Fixed seed 29 · 16 s · stereo · not an isolated accompaniment stem' }),
  );
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
for (const button of document.querySelectorAll('[data-jump]')) button.addEventListener('click', () => document.getElementById(button.dataset.jump).scrollIntoView({ behavior: 'smooth', block: 'start' }));
for (const link of document.querySelectorAll('[data-open-view]')) link.addEventListener('click', (event) => { event.preventDefault(); setView(link.dataset.openView); window.scrollTo({ top: 0, behavior: 'smooth' }); });
window.addEventListener('hashchange', () => setView(location.hash.slice(1)));
setView(location.hash.slice(1) || 'lyrics');
if (window.lucide) window.lucide.createIcons();
try {
  const response = await fetch('./results.json');
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const data = await response.json();
  renderChoral(data.choral);
  renderChoralNotes(data.choral_note);
  $('#seed-range').textContent = [data.choral.runs.frame.f1.frame, ...data.choral_seed_repeats.map((row) => row.frame_f1)].map((score) => fmt(score)).join(' / ');
  renderRewards(data.reward_audit);
  renderLyrics(data);
  renderVocal(data.replays.vocal);
  renderStageRail('#choral-stage-rail', data.replays.choral.stages);
  renderSongEval(data.songeval_audit);
} catch (error) {
  $('#headline-grpo').textContent = 'Data unavailable';
  $('#headline-bce').textContent = 'Data unavailable';
  $('#headline-swap').textContent = 'Data unavailable';
  console.error('Could not load experiment receipts', error);
}
