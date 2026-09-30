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

function renderChoralDemo(data) {
  const canvas = $('#choral-piano-roll');
  const ctx = canvas.getContext('2d');
  const notes = data.notes;
  const voices = ['Soprano', 'Alto', 'Tenor', 'Bass'];
  const colors = { reference: '#176b57', baseline: '#b2473c', grpo: '#2869a3', muscriptor: '#805c25' };
  const all = Object.values(notes).flat();
  const limits = voices.map((_, voice) => {
    const values = all.filter((note) => note.voice === voice).map((note) => note.pitch);
    return [Math.min(...values) - 2, Math.max(...values) + 2];
  });
  const draw = (mode) => {
    ctx.fillStyle = '#ffffff'; ctx.fillRect(0, 0, canvas.width, canvas.height);
    const left = 145, right = 1080, top = 24, lane = 88, duration = 10.2;
    if (mode === 'muscriptor') {
      ctx.fillStyle = '#f4f7f6'; ctx.fillRect(0, top, canvas.width, 352);
      ctx.fillStyle = '#17211e'; ctx.font = 'bold 13px system-ui, sans-serif'; ctx.fillText('Unassigned piano', 16, 48);
      ctx.fillStyle = '#63716b'; ctx.font = '11px system-ui, sans-serif'; ctx.fillText('No SATB labels', 16, 68);
      for (let pitch = 48; pitch <= 76; pitch += 4) {
        const y = 350 - (pitch - 48) / 28 * 290;
        ctx.strokeStyle = '#d7ddd9'; ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(right, y); ctx.stroke();
        ctx.fillStyle = '#63716b'; ctx.fillText(String(pitch), 112, y + 3);
      }
      for (const note of data.muscriptor.estimated_notes) {
        const x = left + note.start / duration * (right - left);
        const width = Math.max(3, (note.end - note.start) / duration * (right - left) - 2);
        const y = 350 - (note.pitch - 48) / 28 * 290;
        ctx.fillStyle = colors.muscriptor; ctx.fillRect(x, y - 4, width, 9);
      }
      $('#choral-example-score').textContent = `${data.muscriptor.predicted_notes} piano-track notes · onset F1 ${fmt(data.muscriptor.note_onset_50ms.f1, 3)} · onset+offset F1 ${fmt(data.muscriptor.note_onset_offset_50ms.f1, 3)} · pitch-only`;
      for (const button of document.querySelectorAll('[data-choral-roll]')) button.setAttribute('aria-pressed', String(button.dataset.choralRoll === mode));
      return;
    }
    for (let voice = 0; voice < 4; voice++) {
      const y = top + voice * lane;
      ctx.fillStyle = voice % 2 ? '#f4f7f6' : '#fbfcfb'; ctx.fillRect(0, y, canvas.width, lane);
      ctx.fillStyle = '#17211e'; ctx.font = 'bold 13px system-ui, sans-serif'; ctx.fillText(voices[voice], 16, y + 35);
      ctx.fillStyle = '#63716b'; ctx.font = '11px system-ui, sans-serif'; ctx.fillText(`${limits[voice][0]}–${limits[voice][1]} MIDI`, 16, y + 55);
      ctx.strokeStyle = '#d7ddd9'; ctx.beginPath(); ctx.moveTo(0, y + lane); ctx.lineTo(canvas.width, y + lane); ctx.stroke();
    }
    for (let second = 0; second <= 10; second += 2) {
      const x = left + second / duration * (right - left);
      ctx.strokeStyle = '#ccd5cf'; ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top + 4 * lane); ctx.stroke();
      ctx.fillStyle = '#5f6b66'; ctx.font = '11px system-ui, sans-serif'; ctx.fillText(`${second}s`, x + 4, 384);
    }
    for (const note of notes[mode]) {
      const [low, high] = limits[note.voice];
      const x = left + note.start / duration * (right - left);
      const width = Math.max(3, (note.end - note.start) / duration * (right - left) - 2);
      const y = top + note.voice * lane + 68 - (note.pitch - low) / (high - low) * 52;
      ctx.fillStyle = colors[mode]; ctx.fillRect(x, y, width, 8);
    }
    const score = data.receipt.metrics[mode];
    $('#choral-example-score').textContent = mode === 'reference'
      ? `${notes.reference.length} reference notes · original synthesis`
      : `${notes[mode].length} predicted notes · onset F1 ${fmt(score.note_onset_50ms.f1, 3)} · onset+offset F1 ${fmt(score.note_onset_offset_50ms.f1, 3)}`;
    for (const button of document.querySelectorAll('[data-choral-roll]')) button.setAttribute('aria-pressed', String(button.dataset.choralRoll === mode));
  };
  for (const button of document.querySelectorAll('[data-choral-roll]')) button.addEventListener('click', () => draw(button.dataset.choralRoll));
  draw('reference');
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
    ['Frame-wise (16 ms grid)', 'frame_16ms', 'track_frame_16ms'],
    ['Pitch + onset (50 ms)', 'note_onset_50ms', 'track_note_onset_50ms'],
    ['Pitch + onset + offset (50 ms minimum)', 'note_onset_offset_50ms', 'track_note_onset_offset_50ms'],
  ];
  appendCells($('#choral-note-table'), rows.map(([label, pitch, track]) => [
    label, fmt(data.before[pitch].f1), fmt(data.before[track].f1), '相同',
  ]));
}

function renderAceChoral() {
  const media = (name) => ({
    audio: `audio/${name}.wav`,
    wave: `visuals/${name}_wave.png`,
    spectrum: `visuals/${name}_spectrum.png`,
  });
  $('#choral-ace-compare').replaceChildren(
    createMediaPanel({ label: 'REFERENCE / SAME SINGERS', title: 'Reference SATB MIDI · ACE Studio', asset: media('choral_ace_reference_short'), details: 'Elirah / Emma / Julian / Mangus · la · 11 s' }),
    createMediaPanel({ label: 'BASELINE / SAME SINGERS', title: 'ChoralStream baseline MIDI · ACE Studio', asset: media('choral_ace_baseline'), details: '同一四位歌手 · frame-head GRPO MIDI 完全相同' }),
  );
}

function drawSatbComparison(canvas, noteSets, stage, label) {
  const voices = ['Soprano', 'Alto', 'Tenor', 'Bass'];
  const ctx = canvas.getContext('2d');
  const allNotes = Object.values(noteSets).flat();
  const ranges = voices.map((_, voice) => {
    const pitches = allNotes.filter((note) => note.voice === voice).map((note) => note.pitch);
    return [Math.min(...pitches) - 1, Math.max(...pitches) + 1];
  });
  const left = 145, right = 1080, top = 25, lane = 86, duration = 10.2;
  ctx.fillStyle = '#ffffff'; ctx.fillRect(0, 0, canvas.width, canvas.height);
  for (let voice = 0; voice < 4; voice++) {
    const y = top + voice * lane;
    ctx.fillStyle = voice % 2 ? '#f4f7f6' : '#fbfcfb'; ctx.fillRect(0, y, canvas.width, lane);
    ctx.fillStyle = '#17211e'; ctx.font = 'bold 13px system-ui, sans-serif'; ctx.fillText(voices[voice], 12, y + 25);
    ctx.fillStyle = '#63716b'; ctx.font = '11px system-ui, sans-serif';
    ctx.fillText('reference', 78, y + 29); ctx.fillText(label, 78, y + 64);
    ctx.strokeStyle = '#d7ddd9'; ctx.beginPath(); ctx.moveTo(0, y + lane); ctx.lineTo(canvas.width, y + lane); ctx.stroke();
  }
  for (let second = 0; second <= 10; second += 2) {
    const x = left + second / duration * (right - left);
    ctx.strokeStyle = '#ccd5cf'; ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top + 4 * lane); ctx.stroke();
    ctx.fillStyle = '#5f6b66'; ctx.font = '11px system-ui, sans-serif'; ctx.fillText(`${second}s`, x + 4, 381);
  }
  for (const [notes, color, yOffset] of [[noteSets.reference, '#176b57', 8], [noteSets[stage], '#b2473c', 43]]) {
    for (const note of notes) {
      const [low, high] = ranges[note.voice];
      const x = left + note.start / duration * (right - left);
      const width = Math.max(2, (note.end - note.start) / duration * (right - left) - 2);
      const y = top + note.voice * lane + yOffset + 23 - 20 * (note.pitch - low) / (high - low);
      ctx.fillStyle = color; ctx.fillRect(x, y, width, 7);
    }
  }
}

function renderEventReplay(data) {
  const steps = [0, 100, 300, 1000];
  const asset = (step) => {
    const id = String(step).padStart(4, '0');
    return { audio: `audio/choral_event_${id}.wav`,
      wave: `visuals/choral_event_${id}_wave.png`,
      spectrum: `visuals/choral_event_${id}_spectrum.png` };
  };
  function select(step) {
    const before = data.receipt.steps['0'];
    const after = data.receipt.steps[String(step)];
    const rail = $('#choral-event-rail');
    rail.replaceChildren(...steps.map((candidate) => {
      const button = document.createElement('button'); button.type = 'button';
      button.className = 'stage-item available'; button.setAttribute('aria-pressed', String(candidate === step));
      const title = document.createElement('strong'); title.textContent = `${candidate} updates`;
      const state = document.createElement('span'); state.className = 'state measured';
      state.textContent = candidate === 0 ? '冻结起点' : '可试听';
      button.append(title, state); button.addEventListener('click', () => select(candidate));
      return button;
    }));
    $('#choral-event-compare').replaceChildren(
      createMediaPanel({ label: 'A / 0 UPDATES', title: 'Frozen ChoralStream', asset: asset(0),
        details: `${before.note_count} notes · fixed singers` }),
      createMediaPanel({ label: `B / ${step} UPDATES`, title: `Combined-reward GRPO · ${step}`, asset: asset(step),
        details: `${after.note_count} notes · same input, singers and export settings` }),
    );
    $('#choral-event-stage-head').textContent = `${step} 步`;
    const stageMidi = $('#choral-event-stage-midi');
    stageMidi.href = `./midi/choral_event_${String(step).padStart(4, '0')}.mid`;
    stageMidi.textContent = `${step} 步 MIDI ↓`;
    const metrics = [
      ['Frame F1', 'frame_f1'], ['Onset F1 · 50 ms', 'onset_f1'],
      ['Onset + offset F1', 'onset_offset_f1'],
      ['SATB onset F1', 'track_onset_f1'], ['SATB complete-note F1', 'track_note_f1'],
    ];
    appendCells($('#choral-event-replay-metrics'), metrics.map(([label, key]) => [
      label, fmt(before[key], 3), fmt(after[key], 3), signed(after[key] - before[key], 3),
    ]));
    drawSatbComparison($('#choral-event-piano-roll'), data.notes, String(step), `${step} steps`);
  }
  select(300);
}

function renderEventPilot(data) {
  const summary = [['Frozen · 0', 0, data.baseline], ...[100, 300, 1000].map((step) => {
    const stage = data.arms.combined.milestones[String(step)];
    return [`Combined · ${step}`, stage.updated_steps, stage.metrics];
  })];
  appendCells($('#choral-event-summary'), summary.map(([label, updates, metrics]) => [
    label, String(updates), ...['frame', 'onset', 'onset_offset'].map((metric) => fmt(metrics.macro[metric], 3)),
  ]));
  const combinedRows = [['Frozen · 0', data.baseline, 0]];
  for (const step of [100, 300, 1000]) {
    const stage = data.arms.combined?.milestones?.[String(step)];
    if (stage) combinedRows.push([`Combined · ${step}`, stage.metrics, stage.updated_steps]);
  }
  function renderRows(step) {
    const rows = [...combinedRows];
    for (const arm of ['onset', 'onset_offset', 'frame', 'coverage', 'continuity', 'weak_voice', 'precision']) {
      const stage = data.arms[arm]?.milestones?.[String(step)];
      if (stage) rows.push([`${arm} only · ${step}`, stage.metrics, stage.updated_steps]);
    }
    appendCells($('#choral-event-table'), rows.map(([label, metrics, updates]) => [
      label, String(updates), fmt(metrics.macro.frame), fmt(metrics.macro.onset), fmt(metrics.macro.onset_offset),
      fmt(metrics.track_onset.f1), fmt(metrics.track_note.f1),
    ]));
    for (const button of document.querySelectorAll('[data-event-arm-step]')) {
      button.setAttribute('aria-pressed', String(Number(button.dataset.eventArmStep) === step));
    }
  }
  for (const button of document.querySelectorAll('[data-event-arm-step]')) {
    button.addEventListener('click', () => renderRows(Number(button.dataset.eventArmStep)));
  }
  renderRows(1000);
  const partRows = [combinedRows[0], ...combinedRows.filter(([label]) => label === 'Combined · 1000' || label === 'Combined · 300')];
  appendCells($('#choral-event-parts-table'), partRows.map(([label, metrics]) => [
    label,
    ...['S', 'A', 'T', 'B'].flatMap((voice) => ['frame', 'onset', 'onset_offset'].map((metric) => fmt(metrics.per_voice[voice][metric].f1, 3))),
    ...['frame', 'onset', 'onset_offset'].map((metric) => fmt(metrics.macro[metric], 3)),
    fmt(metrics.va_rate_percent.frame, 2), fmt(metrics.va_rate_percent.onset, 2),
  ]));
}

function renderActualArmReplays(replay, pilot) {
  const names = { combined: 'Combined', onset: 'Onset only', onset_offset: 'Onset + offset only',
    frame: 'Frame only', coverage: 'Coverage only', continuity: 'Continuity only',
    weak_voice: 'Weak-voice only', precision: 'Precision only' };
  const stageKey = (arm) => arm === 'combined' ? '300' : `arm_${arm}_300`;
  const assetName = (arm) => arm === 'combined' ? 'choral_event_0300' :
    arm === 'weak_voice' ? 'choral_event_0000' : `choral_event_arm_${arm}_300`;
  const asset = (name) => ({ audio: `audio/${name}.wav`,
    wave: `visuals/${name}_wave.png`, spectrum: `visuals/${name}_spectrum.png` });
  const rows = [['Frozen', 0, replay.receipt.steps['0']],
    ...Object.keys(names).map((arm) => [names[arm],
      pilot.arms[arm].milestones['300'].updated_steps,
      replay.receipt.steps[stageKey(arm)]])];
  appendCells($('#actual-arm-table'), rows.map(([label, updates, metrics]) => [
    label, String(updates), String(metrics.note_count), fmt(metrics.frame_f1, 3),
    fmt(metrics.onset_f1, 3), fmt(metrics.onset_offset_f1, 3),
  ]));
  const select = $('#actual-arm-select');
  function render() {
    const arm = select.value;
    const key = stageKey(arm);
    const result = replay.receipt.steps[key];
    const name = assetName(arm);
    $('#actual-arm-midi').href = `./midi/${arm === 'weak_voice' ? 'choral_event_arm_weak_voice_300' : name}.mid`;
    $('#actual-arm-audio').replaceChildren(
      createMediaPanel({ label: 'A / FROZEN', title: 'Baseline · 0 updates',
        asset: asset('choral_event_0000'), details: 'Same original input and fixed singers' }),
      createMediaPanel({ label: `B / ${names[arm].toUpperCase()}`, title: `${names[arm]} · 300 attempts`,
        asset: asset(name), details: `${result.note_count} notes · ${pilot.arms[arm].milestones['300'].updated_steps} effective updates` }),
    );
    drawSatbComparison($('#actual-arm-piano-roll'), replay.notes, key, names[arm]);
  }
  select.addEventListener('change', render);
  render();
}

function renderPawctPaper(data) {
  appendCells($('#pawct-paper-table'), data.rows.map((row) => [
    row.model,
    ...['S', 'A', 'T', 'B', 'average'].flatMap((part) => row[part].map((value) => fmt(value, 3))),
    ...row.va_rate.map((value) => fmt(value, 2)),
  ]));
}

function renderSatbCounterexamples(data) {
  const cases = [
    ['onset_only_short_notes', 'Shorten every note', 'onset'],
    ['overfill_octave', 'Add octave duplicates', 'coverage'],
    ['silence', 'Delete all notes', 'continuity'],
    ['fragment_sustained', 'Split sustained notes', 'frame'],
    ['swap_soprano_alto', 'Swap S/A', 'coverage'],
    ['drop_bass', 'Delete Bass', 'precision'],
  ];
  appendCells($('#satb-counterexamples-table'), cases.map(([key, label, misleading]) => {
    const row = data.variants[key];
    return [label, misleading, fmt(row.components[misleading], 3), fmt(row.combined, 3),
      fmt(row.metrics.macro.onset, 3), fmt(row.metrics.macro.onset_offset, 3)];
  }));
}

function renderSatbBadcaseAudio() {
  const choices = {
    onset_only_short_notes: 'Onset-only short notes',
    overfill_octave: 'Coverage-only octave overfill',
    fragment_sustained: 'Frame-only fragmented notes',
  };
  const asset = (name) => ({ audio: `audio/${name}.wav`,
    wave: `visuals/${name}_wave.png`, spectrum: `visuals/${name}_spectrum.png` });
  const render = () => {
    const key = $('#satb-badcase-select').value;
    $('#satb-badcase-midi').href = `./midi/choral_audit_${key}.mid`;
    $('#satb-badcase-audio').replaceChildren(
      createMediaPanel({ label: 'A / REFERENCE', title: 'Original SATB',
        asset: asset('choral_ace_reference_short'), details: '32 reference notes · fixed voices' }),
      createMediaPanel({ label: 'B / CONSTRUCTED BAD CASE', title: choices[key],
        asset: asset(`choral_audit_${key}`), details: 'Same four singers · not a GRPO output' }),
    );
  };
  $('#satb-badcase-select').addEventListener('change', render);
  render();
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
    state.textContent = stage.status === 'measured' ? '可试听' : stage.status === 'metrics_only' ? '仅指标' : stage.status === 'running' ? '运行中' : '待运行';
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
  const spectrumImage = document.createElement('img'); spectrumImage.src = `./${asset.spectrum}`; spectrumImage.alt = `${title} spectrogram on a logarithmic frequency scale`; spectrumImage.loading = 'lazy';
  const caption = document.createElement('figcaption'); caption.textContent = 'Log-frequency spectrogram';
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
  const audioLink = document.createElement('a');
  audioLink.href = `./${asset.audio}`;
  audioLink.target = '_blank';
  audioLink.rel = 'noopener';
  audioLink.className = 'audio-fallback';
  audioLink.textContent = '单独打开音频 ↗';
  panel.append(index, heading, wave, spectrum, audio, audioLink, note);
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
    const running = descriptor.stages.filter((stage) => stage.status === 'running').map((stage) => stage.step);
    $('#lyrics-stage-context').textContent = `${running.length ? `${running.join(' / ')} steps 正在训练。` : ''}${pending.length ? `${pending.join(' / ')} steps 待运行。` : ''}欠拟合、改善或 reward hacking 必须由留出音频与指标共同判断，不能按步数预设。`;
    document.querySelector('[data-listen="candidate"]').textContent = `B · ${step} ${step === 1 ? 'step' : 'steps'}`;
    $('#lyrics-metric-head').textContent = `${step} ${step === 1 ? 'step' : 'steps'}`;
    const pairedBaseline = selected.paired_baseline || baseline;
    $('#lyrics-compare').replaceChildren(
      createMediaPanel({ label: 'A / BASELINE', title: `${descriptor.name} · 0 updates`, asset: pairedBaseline, details: '48 kHz · held-out seed 5101', listenRole: 'baseline' }),
      createMediaPanel({ label: 'B / GRPO', title: `${descriptor.name} · ${step} ${step === 1 ? 'update' : 'updates'}`, asset: selected, details: '同 prompt、同 seed · 独立留出样本', listenRole: 'candidate' }),
    );
    const before = pairedBaseline.metrics, after = selected.metrics;
    const rewardName = descriptor.reward_model || 'SongEval';
    const metrics = [
      ...['Coherence', 'Musicality', 'Memorability', 'Clarity', 'Naturalness', 'mean'].map((key) => [`${rewardName} · ${key}`, before.reward[key], after.reward[key], 4]),
      ['Audio · peak', before.signal.peak, after.signal.peak, 3],
      ['Audio · RMS', before.signal.rms, after.signal.rms, 3],
      ['Audio · ≥0.999 samples (%)', 100 * before.signal.near_full_scale_fraction, 100 * after.signal.near_full_scale_fraction, 4],
    ];
    appendCells($('#lyrics-metric-table'), metrics.map(([name, a, b, digits]) => [name, fmt(a, digits), fmt(b, digits), signed(b - a, digits)]));
    if (step === 1 && model === 'muse') {
      const changed = `${receipt.heldout_generated_token_difference}/${receipt.max_new_tokens} generated tokens`;
      $('#lyrics-interpretation').textContent = `未训练重放波形完全相同；更新后 ${changed} 改变。本次留出片段 SongEval ${fmt(before.reward.mean)} → ${fmt(after.reward.mean)}，两版均无满幅削波。只有 2 条训练 rollout、1 次优化与 1 条留出样本；不足以判断长程优化、听感改善或 reward hacking。`;
    } else if (model === 'musecritic') {
      $('#lyrics-interpretation').textContent = `MuseCritic GRPO ${step} 步的单条留出音频：均分 ${fmt(before.reward.mean)} → ${fmt(after.reward.mean)}；峰值 ${fmt(before.signal.peak, 3)} → ${fmt(after.signal.peak, 3)}。两版均无满幅削波。每个阶段使用各自的 A/B 解码，MuCodec 重解码存在随机性；相同 baseline token 在 25/50 步评估中也生成不同波形，故不能把单条分差全归因于 adapter。`;
    } else if (step === 1) {
      $('#lyrics-interpretation').textContent = `YuE2 本轮从前一日的一步 LoRA 继续训练；0/1 步音频在同一新推理配置下重放。三条固定留出提示的 SongEval 均分 ${fmt(data.replays.lyrics.yue2.stages[0].mean_reward)} → ${fmt(selected.mean_reward)}；第一个样本 ${fmt(before.reward.mean)} → ${fmt(after.reward.mean)}。不是泛化改善证据。`;
    } else {
      $('#lyrics-interpretation').textContent = `YuE2 ${step} 步：三条固定留出提示的 SongEval 均分 ${fmt(baseline.mean_reward)} → ${fmt(selected.mean_reward)}；当前试听样本 ${fmt(before.reward.mean)} → ${fmt(after.reward.mean)}。主观偏好仍需独立核对，不能只凭 reward 认定改善。`;
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

function renderVocal(replay, reward) {
  renderStageRail('#vocal-stage-rail', replay.stages);
  $('#vocal-compare').replaceChildren(
    createMediaPanel({ label: 'INPUT', title: 'Synthetic vocal guide', asset: replay.input, details: 'Original synthetic asset · 16 s · mono' }),
    createMediaPanel({ label: 'BASELINE / NO GRPO', title: 'ACE-Step 1.5 completion', asset: replay.stages[0], details: 'Fixed seed 29 · 16 s · stereo · not an isolated accompaniment stem' }),
    createMediaPanel({ label: 'BASELINE / NO GRPO', title: 'AnyAccomp · accompaniment', asset: replay.outputs[0], details: 'Fixed seed 29 · 16 s · mono · accompaniment only' }),
    createMediaPanel({ label: 'LISTENING MIX', title: 'AnyAccomp · vocal + accompaniment', asset: replay.outputs[1], details: 'Same guide vocal mixed with generated accompaniment · no GRPO' }),
  );
  $('#vocal-coverage').textContent = `${fmt(100 * reward.reward_diagnostics.coverage, 1)}%`;
  $('#vocal-beat').textContent = fmt(reward.reward_diagnostics.beat_v2.score, 3);
  $('#vocal-beat-count').textContent = reward.reward_diagnostics.beat_v2.reference_beats;
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
  renderChoralDemo(data.choral_demo);
  renderAceChoral();
  renderEventReplay(data.event_replay);
  renderEventPilot(data.event_pilot);
  renderActualArmReplays(data.event_replay, data.event_pilot);
  renderPawctPaper(data.pawct_paper);
  renderSatbCounterexamples(data.satb_reward_counterexamples);
  renderSatbBadcaseAudio();
  $('#seed-range').textContent = [data.choral.runs.frame.f1.frame, ...data.choral_seed_repeats.map((row) => row.frame_f1)].map((score) => fmt(score)).join(' / ');
  renderRewards(data.reward_audit);
  renderLyrics(data);
  renderVocal(data.replays.vocal, data.vocal_reward);
  renderSongEval(data.songeval_audit);
} catch (error) {
  $('#headline-grpo').textContent = 'Data unavailable';
  $('#headline-bce').textContent = 'Data unavailable';
  $('#headline-swap').textContent = 'Data unavailable';
  console.error('Could not load experiment receipts', error);
}
