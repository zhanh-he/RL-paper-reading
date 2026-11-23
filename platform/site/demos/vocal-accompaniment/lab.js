const get = (id) => document.getElementById(id);
const sessionPaths = {control: "audio/manifest.json", real: "audio/anyaccomp/manifest.json"};
const voteKey = "vocal-accompaniment-lab-judgments-v1";
const view = {session: null, take: 0, pair: null, monitor: "mix", bar: 0, loading: false};
const player = {context: null, clockStart: 0, position: 0, run: [], request: 0, clipEnd: null};
const buffers = new Map();
let votes = [];
try {
  const saved = JSON.parse(localStorage.getItem(voteKey) || "[]");
  if (Array.isArray(saved)) votes = saved;
} catch { /* Playback is available when storage is restricted. */ }

function node(tag, data, className) {
  const item = document.createElement(tag);
  if (data) item.textContent = data;
  if (className) item.className = className;
  return item;
}

function take() {
  return view.session.variants[view.take];
}

function meter() {
  return view.session?.bars || [];
}

function currentPosition() {
  const clock = player.context;
  const elapsed = player.run.length ? clock.currentTime - player.clockStart : 0;
  return Math.min(view.session?.duration_s ?? 0, player.position + Math.max(0, elapsed));
}

function pause() {
  player.request += 1;
  player.position = currentPosition();
  for (const source of player.run) { source.stop(); source.disconnect(); }
  player.run = [];
  renderClock();
}

async function requestBuffer(file) {
  if (!buffers.has(file)) {
    const pending = fetch(file).then((response) => {
      if (!response.ok) throw new Error("The audio file could not be loaded.");
      return response.arrayBuffer();
    }).then((data) => player.context.decodeAudioData(data)).catch((error) => {
      buffers.delete(file);
      throw error;
    });
    buffers.set(file, pending);
  }
  return buffers.get(file);
}

function audioFiles() {
  const {base, vocal} = view.session;
  const {file, mix_file: mixFile} = take();
  if (view.monitor === "voice") return [[base + vocal.file, 0.72]];
  if (view.monitor === "backing") return [[base + file, 0.72]];
  if (mixFile) return [[base + mixFile, 0.82]];
  return [[base + vocal.file, 0.72], [base + file, 0.60]];
}

async function play() {
  if (!view.session || view.loading) return;
  const request = ++player.request;
  const duration = view.session.duration_s;
  if (player.position >= duration - 0.02) player.position = 0;
  get("status").textContent = "Loading audio…";
  try {
    await player.context.resume();
    const decoded = await Promise.all(audioFiles().map(async ([file, gain]) => [await requestBuffer(file), gain]));
    if (player.request !== request) return;
    player.clockStart = player.context.currentTime + 0.022;
    for (const [buffer, gain] of decoded) {
      if (player.position >= buffer.duration) continue;
      const source = player.context.createBufferSource();
      const volume = player.context.createGain();
      source.buffer = buffer;
      volume.gain.value = gain;
      source.connect(volume).connect(player.context.destination);
      source.onended = () => volume.disconnect();
      source.start(player.clockStart, player.position);
      player.run.push(source);
    }
    get("status").textContent = "";
    renderClock();
  } catch (error) {
    if (player.request === request) get("status").textContent = error.message;
  }
}

function timeText(value) {
  const tenths = Math.floor(value * 10);
  return `${Math.floor(tenths / 600)}:${String(Math.floor(tenths / 10) % 60).padStart(2, "0")}.${tenths % 10}`;
}

function renderClock() {
  if (!view.session) return;
  const position = currentPosition();
  const duration = view.session.duration_s;
  const playing = player.run.length > 0;
  get("clock").textContent = `${timeText(position)} / ${timeText(duration)}`;
  get("cursor").style.left = `${(position / duration) * 100}%`;
  get("scrub").value = position;
  get("play").setAttribute("aria-label", playing ? "Pause" : "Play");
  get("play").firstElementChild.textContent = playing ? "Ⅱ" : "▶";
}

function seek(seconds, auto = player.run.length > 0) {
  pause();
  player.position = seconds;
  if (auto) void play();
  renderClock();
}

function frame() {
  if (player.run.length) {
    const position = currentPosition();
    const end = player.clipEnd ?? view.session.duration_s;
    if (position >= end - 0.012) {
      pause();
      player.position = end;
      player.clipEnd = null;
    }
    renderClock();
  }
  requestAnimationFrame(frame);
}

function plotTrack(context, samples, top, height, width, color) {
  if (!samples) return;
  context.fillStyle = color;
  const step = samples.length / width;
  for (let column = 0; column < width; column += 2) {
    const start = Math.floor(column * step);
    const end = Math.min(samples.length, Math.floor((column + 2) * step));
    let peak = 0;
    for (let index = start; index < end; index += 3) peak = Math.max(peak, Math.abs(samples[index]));
    const extent = Math.min(height, (Math.sqrt(peak) * height * 1.5));
    context.fillRect(column, top + (height - extent) / 2, 1, Math.max(1, extent));
  }
}

let drawRevision = 0;
async function plot() {
  const revision = ++drawRevision;
  if (!view.session) return;
  const {base, duration_s: duration} = view.session;
  try {
    const [voice, backing] = await Promise.all([
      requestBuffer(base + view.session.vocal.file), requestBuffer(base + take().file),
    ]);
    if (drawRevision !== revision) return;
    const canvas = get("wave");
    const width = Math.max(1, Math.floor(canvas.clientWidth));
    const height = canvas.clientHeight;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = width * ratio;
    canvas.height = height * ratio;
    const context = canvas.getContext("2d");
    context.scale(ratio, ratio);
    const bars = meter();
    if (bars.length) {
      const start = bars[view.bar].start_s / duration * width;
      context.fillStyle = "#242439";
      context.fillRect(start, 0, (bars[view.bar].end_s - bars[view.bar].start_s) / duration * width, height);
    }
    context.fillStyle = "#333546";
    context.fillRect(0, 49, width, 1);
    for (const marker of bars.length ? bars.map((bar) => bar.start_s) : (view.session.vocal.onsets_s || [])) {
      context.fillRect(Math.round(marker / duration * width), 0, 1, height);
    }
    plotTrack(context, voice.getChannelData(0), 3, 41, width, "#d7fb9d");
    if (!view.pair) plotTrack(context, backing.getChannelData(0), 71, 26, width, "#b6a4ff");
  } catch (error) {
    if (drawRevision === revision) get("status").textContent = error.message;
  }
}

function toggleGroup(selector, key, value) {
  for (const button of document.querySelectorAll(selector)) {
    button.setAttribute("aria-pressed", String(button.dataset[key] === value));
  }
}

function renderInspection() {
  const inspection = get("inspection");
  inspection.replaceChildren();
  if (view.pair) {
    inspection.append(node("p", "Switch between A and B at the same playhead position. Listen to both mixes before saving a judgment."));
    return;
  }
  const bar = meter()[view.bar];
  if (!bar) {
    const provenance = view.session.provenance;
    inspection.append(node("strong", provenance?.label || "Model output"));
    inspection.append(node("p", provenance?.detail || "No annotated meter is available for this recording."));
    if (provenance?.url) {
      const link = node("a", "Run provenance");
      link.href = provenance.url;
      const paragraph = node("p");
      paragraph.append(link);
      inspection.append(paragraph);
    }
    return;
  }
  const observed = take().observed_bars?.find((entry) => entry.index === bar.index);
  const planned = take().bar_events?.find((entry) => entry.index === bar.index);
  const line = node("div", "", "inspection-title");
  line.append(node("strong", `Bar ${bar.index} · ${bar.section}`));
  if (observed) {
    const badge = observed.offset_ms === null ? "No accent detected" : `${observed.offset_ms >= 0 ? "+" : ""}${observed.offset_ms} ms`;
    line.append(node("span", badge, `result ${observed.status}`));
  }
  inspection.append(line, node("p", planned?.note || "Listen to the relationship between the backing and vocal here."));
  const playBar = node("button", "Play this bar", "text-button");
  playBar.type = "button";
  playBar.addEventListener("click", () => { player.clipEnd = bar.end_s; seek(bar.start_s, true); });
  inspection.append(playBar);
}

function selectBar(index) {
  view.bar = index;
  player.clipEnd = null;
  seek(meter()[index].start_s);
  renderTimeline();
  renderInspection();
  void plot();
}

function renderTimeline() {
  const timeline = get("timeline");
  const bars = meter();
  const onsets = view.session.vocal.onsets_s || [];
  timeline.replaceChildren();
  timeline.style.setProperty("--bars", String(bars.length || onsets.length || 1));
  if (!bars.length) {
    onsets.forEach((time, index) => {
      const button = node("button", `N${index + 1}`);
      button.type = "button";
      button.setAttribute("aria-label", `Jump to vocal note ${index + 1}`);
      button.addEventListener("click", () => { player.clipEnd = null; seek(time); });
      timeline.append(button);
    });
    return;
  }
  bars.forEach((bar, index) => {
    const button = node("button", `${bar.index}`);
    button.type = "button";
    button.setAttribute("aria-label", `Bar ${bar.index}`);
    button.setAttribute("aria-current", String(index === view.bar));
    if (!view.pair) {
      const observed = take().observed_bars?.find((entry) => entry.index === bar.index);
      if (observed) button.dataset.state = observed.status;
    }
    button.addEventListener("click", () => selectBar(index));
    timeline.append(button);
  });
}

function changeTake(index) {
  if (view.take === index) return;
  const playing = player.run.length > 0;
  pause();
  view.take = index;
  renderSelection();
  if (playing) void play();
}

function renderSelection() {
  const list = get("take-list");
  const shown = view.pair || view.session.variants.map((_, index) => index);
  list.replaceChildren();
  shown.forEach((index, position) => {
    const candidate = view.session.variants[index];
    const button = node("button", "", "take");
    button.type = "button";
    button.setAttribute("aria-pressed", String(index === view.take));
    const copy = node("span");
    copy.append(node("strong", view.pair ? `Take ${position ? "B" : "A"}` : candidate.label));
    if (!view.pair) copy.append(node("small", candidate.description));
    button.append(node("span", "", "swatch"), copy);
    button.addEventListener("click", () => changeTake(index));
    list.append(button);
  });
  get("vote").hidden = !view.pair;
  const instruction = view.session.variants.length > 1 ? "Switch takes without moving the playhead." : "Solo the guide or backing to hear what the model added.";
  get("takes-intro").textContent = view.pair ? "Candidate names and detector scores are hidden." : view.session.takes_note || instruction;
  get("backing-label").textContent = view.pair ? "Backing waveform hidden for blind review" : "Backing";
  get("explore").setAttribute("aria-pressed", String(!view.pair));
  get("blind").setAttribute("aria-pressed", String(Boolean(view.pair)));
  renderTimeline();
  renderInspection();
  void plot();
}

function blind() {
  const count = view.session.variants.length;
  if (count < 2 || view.pair) return;
  pause();
  const a = Math.floor(Math.random() * count);
  const b = (a + 1 + Math.floor(Math.random() * (count - 1))) % count;
  view.pair = [a, b];
  view.take = a;
  view.monitor = "mix";
  player.position = 0;
  player.clipEnd = null;
  toggleGroup("[data-monitor]", "monitor", "mix");
  get("vote").reset();
  renderSelection();
  renderClock();
}

async function loadSession(id) {
  if (view.loading || id === view.session?.id) return;
  view.loading = true;
  pause();
  get("play").disabled = true;
  get("status").textContent = "Loading session…";
  const path = sessionPaths[id];
  try {
    const response = await fetch(path);
    if (!response.ok) throw new Error("This session could not be loaded.");
    const manifest = await response.json();
    if (!manifest.variants?.length || !manifest.vocal?.file || !(manifest.duration_s > 0)) throw new Error("The session manifest is incomplete.");
    player.context ||= new (window.AudioContext || window.webkitAudioContext)();
    view.session = {...manifest, id, base: path.slice(0, path.lastIndexOf("/") + 1)};
    Object.assign(view, {take: 0, pair: null, monitor: "mix", bar: 0});
    player.position = 0;
    player.clipEnd = null;
    get("scrub").max = manifest.duration_s;
    get("session-title").textContent = manifest.title || "Eight bars, one vocal";
    get("session-meta").textContent = manifest.bpm ? `${manifest.bpm} BPM · ${manifest.beats_per_bar}/4 · ~${Math.round(manifest.duration_s)} s · Procedural audio` : `${Math.round(manifest.duration_s)} s · Original synthetic guide · No annotated meter`;
    get("timeline").setAttribute("aria-label", manifest.bars.length ? "Jump to a bar" : "Jump to a vocal note");
    get("status").textContent = "";
    get("blind").disabled = manifest.variants.length < 2;
    toggleGroup("[data-session]", "session", id);
    toggleGroup("[data-monitor]", "monitor", "mix");
    renderSelection();
    renderClock();
  } catch (error) {
    get("status").textContent = error.message;
  } finally {
    get("play").disabled = !view.session;
    view.loading = false;
  }
}

function showVoteCount() {
  get("export").disabled = !votes.length;
  get("saved").textContent = votes.length ? `${votes.length} saved on this device` : "";
}

get("play").addEventListener("click", () => {
  if (player.run.length) pause();
  else void play();
});
get("blind").addEventListener("click", blind);
get("explore").addEventListener("click", () => {
  if (!view.pair) return;
  view.pair = null;
  renderSelection();
});
get("scrub").addEventListener("input", (event) => {
  player.clipEnd = null;
  seek(Number(event.target.value));
});
for (const button of document.querySelectorAll("[data-session]")) {
  button.addEventListener("click", () => { void loadSession(button.dataset.session); });
}
for (const button of document.querySelectorAll("[data-monitor]")) {
  button.addEventListener("click", () => {
    if (button.dataset.monitor === view.monitor) return;
    const playing = player.run.length > 0;
    pause();
    view.monitor = button.dataset.monitor;
    toggleGroup("[data-monitor]", "monitor", view.monitor);
    if (playing) void play();
  });
}
get("vote").addEventListener("submit", (event) => {
  event.preventDefault();
  if (!view.pair) return;
  const fields = new FormData(event.currentTarget);
  const winner = fields.get("winner");
  const candidateIds = view.pair.map((index) => view.session.variants[index].id);
  votes.push({session_id: view.session.id, dataset_version: view.session.version || 1,
    presentation: {A: candidateIds[0], B: candidateIds[1]}, preference: winner,
    chosen_candidate_id: winner === "tie" ? null : candidateIds[winner === "A" ? 0 : 1],
    rejected_candidate_id: winner === "tie" ? null : candidateIds[winner === "A" ? 1 : 0],
    timing_judgment: fields.get("timing_review") || null, reason: fields.get("reason"),
    note: fields.get("note").trim(), recorded_at: new Date().toISOString(),
  });
  let persisted = true;
  try { localStorage.setItem(voteKey, JSON.stringify(votes)); }
  catch { persisted = false; }
  event.currentTarget.reset();
  showVoteCount();
  if (!persisted) get("saved").textContent = "Storage unavailable. Export before you close the page.";
});
get("export").addEventListener("click", () => {
  const payload = new Blob([votes.map((vote) => JSON.stringify(vote)).join("\n"), "\n"], {type: "application/x-ndjson"});
  const url = URL.createObjectURL(payload);
  const link = node("a");
  link.href = url;
  link.download = "accompaniment-listening.jsonl";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
new ResizeObserver(() => { void plot(); }).observe(get("wave"));
showVoteCount();
requestAnimationFrame(frame);
void loadSession("control");
