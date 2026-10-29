const $ = (id) => document.getElementById(id);
const taskLabels = { vocal2accomp: "固定人声 → 伴奏", lyrics2song: "歌词 → 歌曲", extension: "后续扩展", shared: "双任务 / 共享" };
const statusLabels = { proposed: "研究假设", planned: "计划", implemented_unverified: "原型 · 待实测", in_progress: "进行中 · 待回执", verified: "已审核 / 核对", completed: "已完成", failed: "失败", partial: "部分完成" };
const kindLabels = { engineering_check: "工程检查", plan: "任务计划", training_result: "训练实测" };
const statusClasses = { proposed: "mg-proposed", planned: "mg-planned", implemented_unverified: "mg-implemented", in_progress: "mg-progress", verified: "mg-verified", completed: "mg-verified" };
const evidenceLabels = { training_measurement: "训练 / 资源实测", quality_measurement: "音乐质量评测" };
const tabNames = ["research", "records", "sources"];
const sectionTabs = { hypotheses: "research", ablations: "research", roadmap: "research", capabilities: "research" };
let records = [];

function element(tag, className, value) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (value !== undefined && value !== null) node.textContent = String(value);
  return node;
}

function write(id, value) { $(id).textContent = value ?? ""; }
function replace(id, children) { $(id).replaceChildren(...children); }
function badge(status, label) { return element("span", `mg-badge ${statusClasses[status] || ""}`, label || statusLabels[status] || status); }
function safeHttps(value) {
  try { const url = new URL(value); return url.protocol === "https:" ? url.href : null; }
  catch { return null; }
}

function setTab(name, focus = false) {
  if (!tabNames.includes(name)) return;
  for (const key of tabNames) {
    const selected = key === name;
    $(`tab-${key}`).setAttribute("aria-selected", String(selected));
    $(`tab-${key}`).tabIndex = selected ? 0 : -1;
    $(key).hidden = !selected;
  }
  if (focus) $(`tab-${name}`).focus();
}

function followHash() {
  const key = location.hash.slice(1);
  setTab(tabNames.includes(key) ? key : sectionTabs[key] || "research");
  if (tabNames.includes(key) || Object.hasOwn(sectionTabs, key)) requestAnimationFrame(() => $(key).scrollIntoView({ block: "start" }));
}

for (const tab of document.querySelectorAll("[data-tab]")) {
  tab.addEventListener("click", () => {
    const name = tab.dataset.tab;
    setTab(name);
    history.replaceState(null, "", `#${name}`);
  });
  tab.addEventListener("keydown", (event) => {
    const index = tabNames.indexOf(tab.dataset.tab);
    let next;
    if (event.key === "ArrowRight") next = (index + 1) % tabNames.length;
    if (event.key === "ArrowLeft") next = (index + tabNames.length - 1) % tabNames.length;
    if (event.key === "Home") next = 0;
    if (event.key === "End") next = tabNames.length - 1;
    if (next !== undefined) {
      event.preventDefault();
      setTab(tabNames[next], true);
      history.replaceState(null, "", `#${tabNames[next]}`);
    }
  });
}
window.addEventListener("hashchange", followHash);

function renderHypotheses(items) {
  replace("hypothesis-list", items.map((item) => {
    const article = element("article", "mg-hypothesis");
    article.append(element("h3", "", item.title), element("p", "", item.question), element("small", "", item.test));
    return article;
  }));
}

function renderAblations(items) {
  replace("ablation-body", items.map((item) => {
    const row = element("tr", item.score_plan && item.nar ? "mg-combined" : "");
    const name = element("th", "", item.label); name.scope = "row";
    const state = element("td"); state.append(badge(item.status));
    row.append(name, element("td", "", item.score_plan ? "有" : "无"), element("td", "", item.nar ? "有" : "无"), element("td", "", item.question), state);
    return row;
  }));
}

function renderRoadmap(items) {
  replace("roadmap-list", items.map((item) => {
    const row = element("li");
    row.append(element("h3", "", item.title), badge(item.status), element("p", "", item.gate));
    return row;
  }));
}

function renderCapabilities(items) {
  write("capability-count", `${items.length} 个任务 · 原型与扩展分开验收`);
  replace("capability-list", items.map((item, index) => {
    const details = element("details", "mg-capability");
    details.open = index === 0;
    const summary = element("summary");
    const title = element("span", "mg-cap-title", item.title);
    title.append(element("span", "mg-cap-priority", item.priority));
    summary.append(title, badge(item.status));
    const body = element("div", "mg-cap-body");
    const io = element("div", "mg-cap-io");
    for (const [label, value] of [["输入", item.input], ["输出", item.output]]) {
      const part = element("div"); part.append(element("small", "", label), element("span", "", value)); io.append(part);
    }
    body.append(io, element("p", "", item.boundary));
    details.append(summary, body);
    return details;
  }));
}

function renderMetricTable(metrics) {
  const wrap = element("div", "mg-table-wrap");
  wrap.tabIndex = 0;
  const table = element("table", "mg-table");
  const head = element("thead"); const titles = element("tr");
  for (const title of ["指标 ID", "数值", "样本数", "方向 / 范围"]) { const cell = element("th", "", title); cell.scope = "col"; titles.append(cell); }
  head.append(titles); table.append(head);
  const body = element("tbody");
  for (const metric of metrics) {
    const row = element("tr");
    const value = metric.value === null || metric.value === undefined ? "未测" : `${metric.value}${metric.unit ? ` ${metric.unit}` : ""}`;
    row.append(element("td", "", metric.id), element("td", "", value), element("td", "", metric.n ?? "未记录"), element("td", "", [metric.direction, metric.scope].filter(Boolean).join(" / ")));
    body.append(row);
  }
  table.append(body); wrap.append(table); return wrap;
}

function recordNode(record) {
  const article = element("article", "mg-record");
  article.dataset.kind = record.kind;
  const head = element("div", "mg-record-head");
  const info = element("div"); const meta = element("div", "mg-record-meta");
  for (const value of [kindLabels[record.kind] || record.kind, evidenceLabels[record.evidence_kind], record.date, taskLabels[record.task] || record.task]) {
    if (value) meta.append(element("span", "", value));
  }
  info.append(meta, element("h3", "", record.title || record.experiment_id || record.id));
  const verifiedLabel = record.status === "verified" ? (record.kind === "engineering_check" ? "工程已核对" : record.kind === "training_result" ? "实测已审核" : undefined) : undefined;
  head.append(info, badge(record.status, verifiedLabel));
  article.append(head);
  const main = element("div", "mg-record-main"); const text = element("div");
  if (record.scope || record.priority) text.append(element("p", "mg-record-scope", record.scope || record.priority));
  text.append(element("p", "", record.finding || record.summary || record.boundary));
  if (record.limits) text.append(element("p", "mg-record-limits", record.limits));
  main.append(text);
  if (Number.isInteger(record.tests_passed_per_suite)) {
    const count = element("div", "mg-record-count");
    count.append(element("strong", "", record.tests_passed_per_suite), element("small", "", `项 / 套 × ${record.suite_count} 套 · CPU`));
    main.append(count);
  }
  article.append(main);
  if (Array.isArray(record.metrics) && record.metrics.length) article.append(renderMetricTable(record.metrics));
  if (record.kind === "training_result") {
    const identity = [record.config_id && `配置 ${record.config_id}`, record.split_id && `Split ${record.split_id}`, record.seed !== null && record.seed !== undefined ? `Seed ${record.seed}` : null].filter(Boolean);
    if (identity.length) article.append(element("p", "mg-record-limits", identity.join(" · ")));
  }
  if (record.provenance) {
    const details = element("details", "mg-record-note");
    details.append(element("summary", "", "摘要来源与范围"), element("p", "", record.provenance)); article.append(details);
  }
  return article;
}

function renderRecords() {
  const task = $("task-filter").value, kind = $("kind-filter").value, status = $("status-filter").value;
  const selected = records.filter((record) => (task === "all" || record.task === task) && (kind === "all" || record.kind === kind || record.evidence_kind === kind) && (status === "all" || record.status === status));
  replace("record-list", selected.map(recordNode));
  $("filter-empty").hidden = selected.length > 0;
  const counts = selected.reduce((total, record) => { total[record.kind] = (total[record.kind] || 0) + 1; return total; }, {});
  write("filter-summary", `显示 ${selected.length} / ${records.length} 条 · 工程检查 ${counts.engineering_check || 0} · 任务计划 ${counts.plan || 0} · 训练实测 ${counts.training_result || 0}`);
}

function populateFilters() {
  for (const [id, values, labels] of [["task-filter", [...new Set(records.map((record) => record.task))], taskLabels], ["status-filter", [...new Set(records.map((record) => record.status))], statusLabels]]) {
    const select = $(id);
    const first = select.options[0].cloneNode(true);
    select.replaceChildren(first);
    for (const value of values) { const option = element("option", "", labels[value] || value); option.value = value; select.append(option); }
  }
}

function renderReferences(items) {
  write("reference-count", `${items.length} 项公开来源`);
  replace("reference-list", items.map((item) => {
    const row = element("li");
    const href = safeHttps(item.url);
    const title = element(href ? "a" : "strong", "", `${item.title}${href ? " ↗" : ""}`);
    if (href) { title.href = href; title.target = "_blank"; title.rel = "noopener noreferrer"; }
    row.append(title, element("p", "", item.relation)); return row;
  }));
}

function render(data) {
  const { project, protocol, result_policy: policy } = data;
  document.title = `${project.title} | RL Learning`;
  write("project-subtitle", project.subtitle);
  write("project-objective", project.objective);
  write("project-approach", project.approach);
  write("snapshot-note", project.snapshot_note);
  write("snapshot-date", data.updated_at);
  write("engineering-count", data.evidence.length);
  write("training-count", data.training_results.length);
  replace("project-tasks", [["主任务", project.primary_task], ["公开基线", project.secondary_task]].map(([label, task]) => { const node = element("span", "", label); node.append(element("strong", "", task)); return node; }));
  renderHypotheses(data.hypotheses);
  renderAblations(data.ablations);
  renderRoadmap(data.roadmap);
  renderCapabilities(data.capabilities);
  write("registration-note", protocol.registration_note);
  replace("protocol-controls", protocol.controls.map((text) => element("li", "", text)));
  replace("protocol-evaluation", protocol.evaluation.map((text) => element("li", "", text)));
  write("benchmark-note", protocol.benchmark_note);
  write("project-boundary", project.boundary);
  if (data.training_results.length) {
    write("results-state", `${data.training_results.length} 条实测摘要`);
    replace("training-results", data.training_results.map((record) => recordNode({ ...record, kind: "training_result" })));
  } else {
    write("results-state", "等待审核回传");
    const empty = element("div", "mg-empty"); empty.append(element("strong", "", "尚无经审核的 GPU / 音乐质量结果"), element("p", "", policy.empty_message));
    replace("training-results", [empty]);
  }
  records = [...data.evidence.map((record) => ({ ...record, kind: "engineering_check" })), ...data.training_results.map((record) => ({ ...record, kind: "training_result" })), ...data.capabilities.map((record) => ({ ...record, kind: "plan" }))];
  write("record-tab-count", records.length);
  populateFilters(); renderRecords();
  replace("return-flow", policy.workflow.split("→").map((step) => element("li", "", step.trim())));
  write("retained-fields", policy.retained_fields);
  write("excluded-fields", policy.excluded_fields);
  renderReferences(data.references);
  write("footer-date", `公开快照 · ${data.updated_at}`);
  $("page-content").hidden = false;
  $("load-state").hidden = true;
  $("retry-load").hidden = true;
  followHash();
}

function validate(data) {
  if (data?.schema_version !== 1 || !data.project || !data.protocol || !data.result_policy) throw new Error("Unsupported public snapshot");
  for (const key of ["hypotheses", "capabilities", "ablations", "roadmap", "evidence", "training_results", "references"]) if (!Array.isArray(data[key])) throw new Error("Incomplete public snapshot");
  for (const key of ["controls", "evaluation"]) if (!Array.isArray(data.protocol[key])) throw new Error("Incomplete protocol");
  if (typeof data.result_policy.workflow !== "string") throw new Error("Incomplete return policy");
}

async function load() {
  $("retry-load").disabled = true;
  $("load-state").hidden = false;
  try {
    const response = await fetch("./data/experiments.json", { cache: "no-cache", credentials: "same-origin" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json(); validate(data); render(data);
  } catch {
    $("page-content").hidden = true;
    $("load-state").hidden = false;
    $("load-state").classList.add("mg-error");
    $("load-state").replaceChildren(element("strong", "", "暂时无法读取公开研究快照。"), element("p", "", "请重新读取，或直接查看公开 JSON 与实验发布规范。加载失败不代表实验完成或失败。"));
    const link = element("a", "mg-text-link", "公开 JSON 快照 ↗"); link.href = "./data/experiments.json";
    const guide = element("a", "mg-text-link", "实验发布规范 ↗");
    guide.href = "https://github.com/zhanh-he/RL-paper-reading/blob/main/platform/docs/MUSICGEN_EXPERIMENTS.md";
    guide.target = "_blank"; guide.rel = "noopener noreferrer";
    $("load-state").append(link, element("span", "", " · "), guide); $("retry-load").hidden = false;
  } finally { $("retry-load").disabled = false; }
}

for (const id of ["task-filter", "kind-filter", "status-filter"]) $(id).addEventListener("change", renderRecords);
$("reset-filters").addEventListener("click", () => { for (const id of ["task-filter", "kind-filter", "status-filter"]) $(id).value = "all"; renderRecords(); });
$("retry-load").addEventListener("click", load);
await load();
