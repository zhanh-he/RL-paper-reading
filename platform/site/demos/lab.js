const $ = (selector) => document.querySelector(selector);
const fmt = (value, digits = 4) => Number(value).toFixed(digits);
const signed = (value, digits = 3) => `${value >= 0 ? '+' : ''}${fmt(value, digits)}`;
const satbColors = ['#e3484f', '#ea8b27', '#00a4ad', '#315fd2'];
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
  const muscriptorColor = '#805c25';
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
        ctx.fillStyle = muscriptorColor; ctx.fillRect(x, y - 4, width, 9);
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
      ctx.fillStyle = satbColors[note.voice]; ctx.fillRect(x, y, width, 8);
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
  const makeItem = (label, title, file, detail) => {
    const panel = document.createElement('article'); panel.className = 'audio-item';
    const index = document.createElement('span'); index.className = 'audio-index'; index.textContent = label;
    const heading = document.createElement('h3'); heading.textContent = title;
    const audio = document.createElement('audio'); audio.controls = true; audio.preload = 'metadata'; audio.src = `./audio/${file}.wav`;
    const note = document.createElement('small'); note.textContent = detail;
    panel.append(index, heading, audio, note);
    return panel;
  };
  $('#choral-ace-compare').replaceChildren(
    makeItem('REFERENCE / SAME SINGERS', 'Reference SATB MIDI · ACE Studio', 'choral_ace_reference_short', 'Elirah / Emma / Julian / Mangus · la'),
    makeItem('BASELINE / SAME SINGERS', 'ChoralStream baseline MIDI · ACE Studio', 'choral_ace_baseline', 'Frame-head GRPO MIDI 与此相同'),
  );
}

function drawVoiceMidi(canvas, notes, pitchRange) {
  const ctx = canvas.getContext('2d');
  ctx.setTransform(2, 0, 0, 2, 0, 0);
  const width = 560, height = 310, left = 54, right = 548, top = 12, bottom = 284, duration = 10.2;
  const [low, high] = pitchRange;
  const pitchHeight = (bottom - top) / (high - low + 1);
  ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, width, height);
  for (let pitch = low; pitch <= high; pitch++) {
    const y = top + (high - pitch) * pitchHeight;
    const blackKey = [1, 3, 6, 8, 10].includes(pitch % 12);
    ctx.fillStyle = pitch % 12 === 0 ? '#edf3f0' : blackKey ? '#f5f7f6' : '#fff';
    ctx.fillRect(left, y, right - left, pitchHeight);
    ctx.fillStyle = blackKey ? '#c6d0cb' : '#fff';
    ctx.fillRect(0, y, left - 3, pitchHeight);
    ctx.strokeStyle = pitch % 12 === 0 ? '#aabbb3' : '#e5ebe8';
    ctx.beginPath(); ctx.moveTo(0, y + pitchHeight); ctx.lineTo(right, y + pitchHeight); ctx.stroke();
    if (pitch % 12 === 0) {
      ctx.fillStyle = '#374840'; ctx.font = '700 10px system-ui, sans-serif';
      ctx.fillText(`C${Math.floor(pitch / 12) - 1}`, 7, y + pitchHeight / 2 + 3);
    }
  }
  for (let second = 0; second <= 10; second += 2) {
    const x = left + second / duration * (right - left);
    ctx.strokeStyle = '#c8d4ce'; ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, bottom); ctx.stroke();
    ctx.fillStyle = '#5f6b66'; ctx.font = '10px system-ui, sans-serif'; ctx.fillText(`${second}s`, x + 2, height - 8);
  }
  for (const note of notes) {
    if (!Number.isInteger(note.voice) || note.voice < 0 || note.voice > 3) continue;
    const x = left + note.start / duration * (right - left);
    const noteWidth = Math.max(2, (note.end - note.start) / duration * (right - left) - 1);
    const y = top + (high - note.pitch + 0.5) * pitchHeight;
    const barHeight = Math.max(4, pitchHeight - 2);
    ctx.fillStyle = satbColors[note.voice]; ctx.fillRect(x, y - barHeight / 2, noteWidth, barHeight);
    ctx.strokeStyle = '#20312a'; ctx.lineWidth = 0.6;
    ctx.strokeRect(x + 0.3, y - barHeight / 2 + 0.3, noteWidth - 0.6, barHeight - 0.6);
  }
}

function createChoralMidiPanel(letter) {
  const panel = document.createElement('article'); panel.className = 'choral-midi-panel';
  const heading = document.createElement('div'); heading.className = 'choral-midi-heading';
  const titleBlock = document.createElement('div');
  const title = document.createElement('h4');
  const detail = document.createElement('small');
  titleBlock.append(title, detail);
  if (letter) {
    const badge = document.createElement('span'); badge.className = 'choral-panel-letter'; badge.textContent = letter;
    heading.append(badge);
  }
  heading.append(titleBlock);
  const canvas = document.createElement('canvas'); canvas.width = 1120; canvas.height = 620;
  canvas.setAttribute('role', 'img');
  const pending = document.createElement('div'); pending.className = 'choral-midi-pending'; pending.hidden = true;
  const audio = document.createElement('audio'); audio.controls = true; audio.preload = 'metadata';
  const links = document.createElement('div'); links.className = 'choral-midi-links';
  const midi = document.createElement('a'); midi.className = 'source-link'; midi.download = '';
  const direct = document.createElement('a'); direct.className = 'source-link'; direct.target = '_blank'; direct.rel = 'noreferrer'; direct.textContent = '音频直链 ↗';
  links.append(midi, direct); panel.append(heading, canvas, pending, audio, links);
  return { panel, title, detail, canvas, pending, audio, links, midi, direct };
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

function renderEventReplay(data, pilot) {
  const steps = [1, 100, 300, 1000];
  const armNames = { onset: 'Onset only', onset_offset: 'Onset + offset only', frame: 'Frame only',
    coverage: 'Coverage only', continuity: 'Continuity only' };
  const allNotes = Object.values(data.notes).flat();
  const pitchRange = [Math.min(...allNotes.map((note) => note.pitch)) - 2,
    Math.max(...allNotes.map((note) => note.pitch)) + 2];
  const panels = { baseline: createChoralMidiPanel(null), combined: createChoralMidiPanel('A') };
  const singleArms = Object.keys(armNames);
  const singlePanels = Object.fromEntries(singleArms.map((arm, index) =>
    [arm, createChoralMidiPanel(String.fromCharCode(66 + index))]));
  $('#choral-baseline-panel').replaceChildren(panels.baseline.panel);
  $('#choral-grpo-grid').append(panels.combined.panel,
    ...singleArms.map((arm) => singlePanels[arm].panel));
  drawVoiceMidi($('#choral-reference-midi'), data.notes.reference, pitchRange);
  const updatePanel = (panel, title, detail, notes, stem, midiStem) => {
    panel.title.textContent = title;
    panel.detail.textContent = detail;
    panel.canvas.hidden = false;
    panel.pending.hidden = true;
    panel.audio.hidden = false;
    panel.links.hidden = false;
    panel.canvas.setAttribute('aria-label', `${title} SATB MIDI piano roll`);
    drawVoiceMidi(panel.canvas, notes, pitchRange);
    const audioPath = `./audio/${stem}.wav`;
    if (!panel.audio.src.endsWith(`/${stem}.wav`)) panel.audio.src = audioPath;
    panel.direct.href = audioPath;
    panel.midi.href = `./midi/${midiStem}.mid`;
    panel.midi.textContent = '下载 MIDI ↓';
  };
  const clearPanel = (panel, title, step, metricsAvailable) => {
    panel.title.textContent = title;
    panel.detail.textContent = metricsAvailable ? `${step} 步 · 测试集指标已测` : `${step} 步 · 尚无回放`;
    panel.canvas.hidden = true;
    panel.pending.hidden = false;
    panel.pending.textContent = metricsAvailable ? '该步尚无公开样本 MIDI / 音频' : '该步尚无 MIDI / 音频';
    panel.audio.pause();
    panel.audio.removeAttribute('src');
    panel.audio.load();
    panel.audio.hidden = true;
    panel.links.hidden = true;
    panel.direct.removeAttribute('href');
    panel.midi.removeAttribute('href');
  };
  updatePanel(panels.baseline, 'Frozen ChoralStream · baseline', `${data.receipt.steps['0'].note_count} notes · 0 步`,
    data.notes['0'], 'choral_event_0000', 'choral_event_0000');
  const selectStep = (step) => {
    const id = String(step).padStart(4, '0');
    const result = data.receipt.steps[String(step)];
    updatePanel(panels.combined, 'Combined rewards', `${result.note_count} notes · ${step} 步`,
      data.notes[String(step)], `choral_event_${id}`, `choral_event_${id}`);
    for (const arm of singleArms) {
      const key = `arm_${arm}_${step}`;
      const replay = data.receipt.steps[key];
      const notes = data.notes[key];
      if (replay && notes) {
        const updates = pilot.arms[arm].milestones[String(step)].updated_steps;
        updatePanel(singlePanels[arm], armNames[arm], `${replay.note_count} notes · ${updates}/${step} 有效更新`,
          notes, `choral_event_arm_${arm}_${step}`, `choral_event_arm_${arm}_${step}`);
      } else {
        clearPanel(singlePanels[arm], armNames[arm], step, Boolean(pilot.arms[arm].milestones[String(step)]));
      }
    }
    for (const button of document.querySelectorAll('#choral-event-rail button')) {
      button.setAttribute('aria-pressed', String(Number(button.dataset.step) === step));
    }
  };
  $('#choral-event-rail').replaceChildren(...steps.map((step) => {
    const button = document.createElement('button'); button.type = 'button';
    button.className = 'stage-item available'; button.dataset.step = String(step);
    const title = document.createElement('strong'); title.textContent = `${step} 步`;
    const state = document.createElement('span'); state.className = 'state measured';
    state.textContent = step === 300 ? 'A–F 可试听' : 'A 可试听';
    button.append(title, state); button.addEventListener('click', () => selectStep(step));
    return button;
  }));
  $('#view-choral').addEventListener('play', (event) => {
    if (event.target.tagName !== 'AUDIO') return;
    for (const audio of document.querySelectorAll('#view-choral audio')) {
      if (audio !== event.target) audio.pause();
    }
  }, true);
  selectStep(300);
}

function renderEventPilot(data) {
  const rows = [{ label: 'Frozen', step: 0, updates: 0, metrics: data.baseline, kind: 'baseline' }];
  for (const step of [1, 100, 300, 1000]) {
    const stage = data.arms.combined?.milestones?.[String(step)];
    if (stage) rows.push({ label: 'Combined', step, updates: stage.updated_steps, metrics: stage.metrics, kind: 'combined' });
  }
  const armNames = { onset: 'Onset only', onset_offset: 'Onset + offset only', frame: 'Frame only',
    coverage: 'Coverage only', continuity: 'Continuity only' };
  for (const [arm, label] of Object.entries(armNames)) {
    for (const step of [300, 1000]) {
      const stage = data.arms[arm]?.milestones?.[String(step)];
      if (stage) rows.push({ label, step, updates: stage.updated_steps, metrics: stage.metrics, kind: 'single' });
    }
  }
  const baseline = data.baseline.macro;
  $('#choral-event-total').replaceChildren(...rows.map(({ label, step, updates, metrics, kind }) => {
    const tr = document.createElement('tr'); tr.className = `choral-result-${kind}`;
    if (kind === 'single' && label === 'Onset only' && step === 300) tr.classList.add('choral-result-group-start');
    for (const value of [label, String(step), String(updates)]) {
      const td = document.createElement('td'); td.textContent = value; tr.append(td);
    }
    for (const metric of ['frame', 'onset', 'onset_offset']) {
      const value = metrics.macro[metric];
      const delta = value - baseline[metric];
      const roundedDelta = Math.round(delta * 1000) / 10;
      const td = document.createElement('td'); td.className = 'choral-score-cell';
      const number = document.createElement('strong'); number.textContent = `${fmt(value * 100, 1)}%`;
      const change = document.createElement('small'); change.className = roundedDelta > 0 ? 'positive' : roundedDelta < 0 ? 'negative' : 'neutral';
      change.textContent = `(${roundedDelta === 0 ? '0.0' : signed(roundedDelta, 1)} pp)`;
      td.append(number, change); tr.append(td);
    }
    return tr;
  }));
  const partRows = [rows[0], ...rows.filter((row) => row.kind === 'combined' && [300, 1000].includes(row.step))];
  const percent = (value) => `${fmt(value * 100, 1)}%`;
  const frameOnset = (frame, onset) => `${percent(frame)} (${percent(onset)})`;
  appendCells($('#choral-event-parts-table'), partRows.map(({ label, step, metrics }) => [
    `${label} · ${step}`,
    ...['S', 'A', 'T', 'B'].flatMap((voice) => [
      frameOnset(metrics.per_voice[voice].frame.f1, metrics.per_voice[voice].onset.f1),
      percent(metrics.per_voice[voice].onset_offset.f1),
    ]),
    frameOnset(metrics.macro.frame, metrics.macro.onset), percent(metrics.macro.onset_offset),
    `${fmt(metrics.va_rate_percent.frame, 1)}% (${fmt(metrics.va_rate_percent.onset, 1)}%)`,
  ]));
}

function renderActualArmReplays(replay, pilot) {
  const names = { combined: 'Combined', onset: 'Onset only', onset_offset: 'Onset + offset only',
    frame: 'Frame only', coverage: 'Coverage only', continuity: 'Continuity only' };
  const stageKey = (arm) => arm === 'combined' ? '300' : `arm_${arm}_300`;
  const assetName = (arm) => arm === 'combined' ? 'choral_event_0300' :
    `choral_event_arm_${arm}_300`;
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
    $('#actual-arm-midi').href = `./midi/${name}.mid`;
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
  const percent = (value, digits = 1) => `${fmt(value * 100, digits)}%`;
  const paired = ([frame, onset], digits = 1) => `${percent(frame, digits)} (${percent(onset, digits)})`;
  appendCells($('#pawct-paper-table'), data.rows.map((row) => [
    row.model,
    ...['S', 'A', 'T', 'B', 'average'].flatMap((part) => [paired(row[part]), '—']),
    `${fmt(row.va_rate[0], 2)}% (${fmt(row.va_rate[1], 2)}%)`,
  ]));
}

function renderMuscriptorSmoke(data) {
  const rows = [
    ['训练同歌 · 5 s', data.train.before.metrics, data.train.after.metrics],
    ['异歌 8 首 · 每首 5 s', data.disjoint.before_macro, data.disjoint.after_macro],
  ];
  appendCells($('#muscriptor-smoke-table'), rows.map(([scope, before, after]) => [
    scope,
    ...['frame', 'onset', 'offset'].flatMap((key) => [fmt(before[key], 3), fmt(after[key], 3)]),
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
    state.textContent = stage.status === 'measured' ? '可试听' : stage.status === 'metrics_only' ? '仅指标' : stage.status === 'running' ? '运行中' : stage.status === 'queued' ? '排队中' : '待运行';
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
  const waveCaption = document.createElement('div'); waveCaption.className = 'visual-caption';
  waveCaption.textContent = 'Waveform · amplitude -1 to +1 (fixed)';
  const spectrum = document.createElement('figure'); spectrum.className = 'spectrum-visual';
  const spectrumImage = document.createElement('img'); spectrumImage.src = `./${asset.spectrum}`; spectrumImage.alt = `${title} mel spectrogram with frequency ticks`; spectrumImage.loading = 'lazy';
  const caption = document.createElement('figcaption'); caption.textContent = 'Mel frequency · up to 24 kHz (source Nyquist if lower) · viridis -80 to 0 dB (fixed)';
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
  panel.append(index, heading, wave, waveCaption, spectrum, audio, audioLink, note);
  return panel;
}

function renderYuE2Pairs(pairs) {
  $('#lyrics-100-pair-list').replaceChildren(...pairs.map((pair) => {
    const row = document.createElement('section'); row.className = 'lyrics-pair-row';
    const heading = document.createElement('h4'); heading.textContent = `留出 ${pair.index + 1} · ${pair.prompt.style}`;
    const lyrics = document.createElement('pre'); lyrics.textContent = pair.prompt.lyrics;
    const score = document.createElement('p'); score.className = 'lyrics-pair-score';
    score.textContent = `SongEval Mean5 · ${fmt(pair.baseline_score, 4)} → ${fmt(pair.after_score, 4)} (${signed(pair.after_score - pair.baseline_score, 4)}) · seed ${pair.seed}`;
    const audioRow = document.createElement('div'); audioRow.className = 'lyrics-pair-audio';
    for (const [label, path] of [['A · 0 步', pair.baseline], ['B · 100 步', pair.after]]) {
      const item = document.createElement('div');
      const title = document.createElement('strong'); title.textContent = label;
      const audio = document.createElement('audio'); audio.controls = true; audio.preload = 'metadata'; audio.src = `./${path}`;
      audio.addEventListener('play', () => {
        for (const other of document.querySelectorAll('audio')) if (other !== audio) other.pause();
      });
      const link = document.createElement('a'); link.href = `./${path}`; link.target = '_blank';
      link.rel = 'noopener'; link.textContent = '单独打开 ↗';
      item.append(title, audio, link); audioRow.append(item);
    }
    row.append(heading, lyrics, score, audioRow);
    return row;
  }));
}

function renderLyrics(data) {
  const models = data.replays.lyrics;
  renderYuE2Pairs(models.yue2_stress.pairs_100);
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
    const queued = descriptor.stages.filter((stage) => stage.status === 'queued').map((stage) => stage.step);
    const running = descriptor.stages.filter((stage) => stage.status === 'running').map((stage) => stage.step);
    $('#lyrics-stage-context').textContent = `${running.length ? `${running.join(' / ')} steps 正在训练。` : ''}${queued.length ? `${queued.join(' / ')} steps 已提交 Gadi 排队。` : ''}${pending.length ? `${pending.join(' / ')} steps 待运行。` : ''}欠拟合、改善或 reward hacking 必须由留出音频与指标共同判断，不能按步数预设。`;
    const isYuE2 = model.startsWith('yue2');
    $('#lyrics-heldout-section').hidden = !isYuE2;
    $('#lyrics-100-pairs').hidden = model !== 'yue2_stress';
    $('#lyrics-reward-title').textContent = isYuE2 ? '优化目标 · SongEval 五维均值' :
      model === 'musecritic' ? '优化目标 · MuseCritic Mean5' : '优化目标 · SongEval 五维均值';
    $('#lyrics-reward-definition').textContent = isYuE2 || model === 'muse'
      ? 'R = (Coherence + Musicality + Memorability + Clarity + Naturalness) / 5。它评估音频审美；coverage、乐器 richness、beat 和歌词匹配都没有作为独立 reward，完整时长或削波也没有显式约束。'
      : 'MuseCritic 的五项审美分数取均值，与 SongEval 不是同一量表；词句准确度和音频故障仍需独立检查。';
    $('#lyrics-train-protocol').textContent = isYuE2
      ? `YuE2：8 条原创训练提示，每步同提示采样 2 首并按组内均值/标准差求优势；LoRA、AdamW ${model === 'yue2_stress' ? '1e-4（压力测试）' : '2e-5（常规对照）'}、600 semantic tokens。每组只更新一次，因此 ratio 裁剪在该次梯度中不起作用；本轮无显式 KL、歌词匹配或响度约束。另有 3 条不参与训练的固定提示。`
      : model === 'musecritic' ? 'MuseCritic：独立的 Muse + MuCodec 在线训练；25/50 步实验从 100 条公开提示采样。解码有随机性，单条 A/B 不能直接归因于参数更新。' :
        'Muse：两条训练 rollout 的一步 SongEval-GRPO 工程试验；这里以与 YuE2 相同的风格、歌词和 seed 5101 重新生成留出试听，MuCodec 解码仍有随机性。';
    if (isYuE2) {
      const baseMean = baseline.mean_reward;
      const armRows = [
        ...models.yue2.stages.filter((stage) => stage.status === 'measured').map((stage) => ({ stage, arm: stage.step < 2 ? 'shared' : '2e-5' })),
        ...models.yue2_stress.stages.filter((stage) => stage.status === 'measured' && stage.step >= 2).map((stage) => ({ stage, arm: '1e-4' })),
      ].sort((a, b) => a.stage.step - b.stage.step || (a.arm === '2e-5' ? -1 : 1));
      appendCells($('#lyrics-heldout-table'), armRows.map(({ stage, arm }) => [
        arm, String(stage.step), `${fmt(stage.mean_reward)} (${signed(stage.mean_reward - baseMean, 4)})`,
        fmt(stage.heldout_summary.max_peak, 4), `${stage.heldout_summary.near_full_scale_clips} / ${stage.heldout_summary.n}`,
        `${stage.heldout_summary.truncated} / ${stage.heldout_summary.n}`,
      ]));
    }
    document.querySelector('[data-listen="candidate"]').textContent = `B · ${step} ${step === 1 ? 'step' : 'steps'}`;
    $('#lyrics-metric-head').textContent = `${step} ${step === 1 ? 'step' : 'steps'}`;
    $('#lyrics-compare').replaceChildren(
      createMediaPanel({ label: 'A / FROZEN BASELINE', title: `${descriptor.name} · 0 updates`, asset: baseline, details: '固定音频 · 所有训练阶段共用 · held-out seed 5101', listenRole: 'baseline' }),
      createMediaPanel({ label: 'B / GRPO', title: `${descriptor.name} · ${step} ${step === 1 ? 'update' : 'updates'}`, asset: selected, details: '同 prompt、同 seed · 独立留出样本', listenRole: 'candidate' }),
    );
    const before = baseline.metrics, after = selected.metrics;
    const rewardName = descriptor.reward_model || 'SongEval';
    const metrics = [
      ...['Coherence', 'Musicality', 'Memorability', 'Clarity', 'Naturalness', 'mean'].map((key) => [`${rewardName} · ${key}`, before.reward[key], after.reward[key], 4]),
      ['Audio · peak', before.signal.peak, after.signal.peak, 3],
      ['Audio · RMS', before.signal.rms, after.signal.rms, 3],
      ['Audio · ≥0.999 samples (%)', 100 * before.signal.near_full_scale_fraction, 100 * after.signal.near_full_scale_fraction, 4],
    ];
    appendCells($('#lyrics-metric-table'), metrics.map(([name, a, b, digits]) => [name, fmt(a, digits), fmt(b, digits), signed(b - a, digits)]));
    if (step === 1 && model === 'muse') {
      $('#lyrics-interpretation').textContent = `相同提示词与 seed 5101 的重新生成片段，SongEval ${fmt(before.reward.mean)} → ${fmt(after.reward.mean)}；两版均无满幅削波。B 的峰值与 RMS 也改变，因此这一个样本的分数上涨不能单独证明审美改善。训练只有 2 条 rollout 和 1 次优化，仍需多首歌、独立解码和盲听验证。`;
    } else if (model === 'musecritic') {
      $('#lyrics-interpretation').textContent = `MuseCritic GRPO ${step} 步的单条留出音频：固定 A 的均分 ${fmt(before.reward.mean)} → B 的 ${fmt(after.reward.mean)}；峰值 ${fmt(before.signal.peak, 3)} → ${fmt(after.signal.peak, 3)}。两版均无满幅削波。A 始终是同一文件；B 在各阶段单独解码，MuCodec 的随机性仍使这组试听和分差不能单独归因于 adapter。原始训练 receipt 还保留了每阶段重解码的 paired baseline 供核查。`;
    } else if (model === 'yue2_stress' && step === 5) {
      $('#lyrics-interpretation').textContent = `高 LR 第 5 步三条留出 SongEval 均分 ${fmt(baseline.mean_reward)} → ${fmt(selected.mean_reward)}，低于常规 LR 同步数。试听样本有 12 个近满幅采样（最长连续 4 个），其余两条没有。这是稀疏峰值异常；reward 没有上升，尚不能称为 reward hacking。`;
    } else if (model === 'yue2_stress' && step === 25) {
      $('#lyrics-interpretation').textContent = `高 LR 第 25 步三条留出 SongEval 均分 ${fmt(baseline.mean_reward)} → ${fmt(selected.mean_reward)}，仍低于冻结基线。试听样本有 37 个近满幅采样（最长连续 17 个），其余两条没有。较第 5 步峰值异常更明显，但音频内容也改变；没有证据证明 SongEval 在奖励削波。`;
    } else if (model === 'yue2_stress' && step === 100) {
      $('#lyrics-interpretation').textContent = `高 LR 第 100 步三条留出 SongEval 均分 ${fmt(baseline.mean_reward)} → ${fmt(selected.mean_reward)}，且三首各自都低于起点；常规 LR 第 100 步为 ${fmt(models.yue2.stages.find((stage) => stage.step === 100).mean_reward)}。试听样本的 VAE 原始峰值 ${fmt(selected.probe.vae_preclamp_peak)}、${selected.probe.clamped_samples} 个样本越界，YuE2 管线将其钳到 1；重渲染 FLAC 与公开试听逐文件一致。分数并未升高，这不是已证实的 reward hacking。`;
    } else if (step === 1 && isYuE2) {
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
  const response = await fetch('./results.json', { cache: 'no-store' });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const data = await response.json();
  renderChoral(data.choral);
  renderChoralNotes(data.choral_note);
  renderChoralDemo(data.choral_demo);
  renderAceChoral();
  renderEventReplay(data.event_replay, data.event_pilot);
  renderEventPilot(data.event_pilot);
  renderActualArmReplays(data.event_replay, data.event_pilot);
  renderPawctPaper(data.pawct_paper);
  renderMuscriptorSmoke(data.muscriptor_medium_smoke);
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
