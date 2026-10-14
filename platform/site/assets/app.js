const DATA_URL = "./data/literature.json";
const CSV_HEADERS = [
  "year",
  "venue",
  "other_rating",
  "zhanh_rating",
  "domain",
  "workstream",
  "title",
  "keywords",
  "model_size",
  "model_params_b",
  "generation_speed",
  "measurement_context",
  "model_size_url",
  "generation_speed_url",
  "paper_url",
  "github_url",
  "demo_url",
  "web_url",
  "zhanh_note",
  "felix_note",
  "other_note",
  "hanyu_note",
  "author_org",
  "author_org_url"
];

const elements = {
  search: document.querySelector("#search-input"),
  domain: document.querySelector("#domain-filter"),
  domainMenu: document.querySelector("#domain-menu"),
  domainSummary: document.querySelector("#domain-summary"),
  year: document.querySelector("#year-filter"),
  venue: document.querySelector("#venue-filter"),
  zhanh: document.querySelector("#zhanh-filter"),
  other: document.querySelector("#other-filter"),
  organization: document.querySelector("#organization-filter"),
  workstream: document.querySelector("#workstream-filter"),
  clear: document.querySelector("#clear-filters"),
  copy: document.querySelector("#copy-view"),
  export: document.querySelector("#export-csv"),
  body: document.querySelector("#catalog-body"),
  tableState: document.querySelector("#table-state"),
  resultCount: document.querySelector("#result-count"),
  activeFilters: document.querySelector("#active-filters"),
  syncStatus: document.querySelector("#sync-status"),
  dialog: document.querySelector("#detail-dialog"),
  dialogClose: document.querySelector("#close-dialog"),
  detailMeta: document.querySelector("#detail-meta"),
  detailTitle: document.querySelector("#detail-title"),
  detailLabels: document.querySelector("#detail-labels"),
  detailKeywords: document.querySelector("#detail-keywords"),
  detailNotes: document.querySelector("#detail-notes"),
  detailMetrics: document.querySelector("#detail-metrics"),
  detailOrganization: document.querySelector("#detail-organization"),
  detailLinks: document.querySelector("#detail-links"),
  toast: document.querySelector("#toast")
};

const state = {
  catalog: [],
  filtered: [],
  excludedDomains: new Set(),
  filters: {
    search: "",
    domain: "",
    year: "",
    venue: "",
    workstream: "",
    zhanh: "",
    other: "",
    organization: ""
  },
  sort: "source_order",
  direction: "asc"
};

const collator = new Intl.Collator(["en", "zh-CN"], {
  numeric: true,
  sensitivity: "base"
});

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function refreshIcons() {
  if (window.lucide) window.lucide.createIcons();
}

function optionValues(key, numeric = false) {
  const values = [...new Set(state.catalog.map((record) => record[key]).filter(Boolean))];
  return values.sort((left, right) => {
    if (!numeric) return collator.compare(left, right);
    const leftValue = key.includes("rating") ? ratingValue(left) : Number(left);
    const rightValue = key.includes("rating") ? ratingValue(right) : Number(right);
    return rightValue - leftValue;
  });
}

function populateSelect(select, values) {
  const current = select.value;
  for (const value of values) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    select.append(option);
  }
  select.value = current;
}

function hydrateFilters() {
  const domains = optionValues("domain");
  state.excludedDomains = new Set([...state.excludedDomains].filter((value) => domains.includes(value)));
  if (state.filters.domain) {
    const included = state.filters.domain === "MusicGen"
      ? domains.filter((domain) => domain.startsWith("MusicGen"))
      : [state.filters.domain];
    if (included.some((domain) => domains.includes(domain))) {
      state.excludedDomains = new Set(domains.filter((domain) => !included.includes(domain)));
    }
    state.filters.domain = "";
  }
  for (const domain of domains) {
    const label = el("label", "domain-option");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.dataset.domain = domain;
    input.setAttribute("aria-label", domain);
    const count = state.catalog.filter((record) => record.domain === domain).length;
    label.append(input, el("span", "", domain), el("small", "", String(count)));
    elements.domain.append(label);
  }
  populateSelect(elements.year, optionValues("year", true));
  populateSelect(elements.venue, optionValues("venue"));
  populateSelect(elements.zhanh, optionValues("zhanh_rating", true));
  populateSelect(elements.other, optionValues("other_rating", true));
  populateSelect(elements.organization, optionValues("author_org"));
}

function readUrlState() {
  const params = new URLSearchParams(window.location.search);
  for (const key of Object.keys(state.filters)) {
    state.filters[key] = params.get(key) ?? (key === "other" ? params.get("felix") : null) ?? "";
  }
  state.sort = params.get("sort") || "source_order";
  if (state.sort === "felix_rating") state.sort = "other_rating";
  state.direction = params.get("dir") === "desc" ? "desc" : "asc";
  state.excludedDomains = new Set(params.getAll("exclude"));
}

function syncControls() {
  elements.search.value = state.filters.search;
  for (const checkbox of elements.domain.querySelectorAll("input")) {
    checkbox.checked = !state.excludedDomains.has(checkbox.dataset.domain);
  }
  const total = optionValues("domain").length;
  const visible = total - state.excludedDomains.size;
  elements.domainSummary.textContent = state.excludedDomains.size ? `${visible} / ${total} domains` : "All domains";
  elements.year.value = state.filters.year;
  elements.venue.value = state.filters.venue;
  elements.zhanh.value = state.filters.zhanh;
  elements.other.value = state.filters.other;
  elements.organization.value = state.filters.organization;
  for (const button of elements.workstream.querySelectorAll("button")) {
    button.setAttribute(
      "aria-pressed",
      String(button.dataset.value === state.filters.workstream)
    );
  }
}

function updateUrl() {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(state.filters)) {
    if (value) params.set(key, value);
  }
  for (const domain of [...state.excludedDomains].sort(collator.compare)) params.append("exclude", domain);
  if (state.sort !== "source_order") params.set("sort", state.sort);
  if (state.direction !== "asc") params.set("dir", state.direction);
  const query = params.toString();
  window.history.replaceState(null, "", `${window.location.pathname}${query ? `?${query}` : ""}`);
}

function ratingValue(value) {
  if (!value) return -1;
  return Number.parseFloat(value.split("/")[0]);
}

function recordSearchText(record) {
  return [
    record.title,
    record.keywords,
    record.model_size,
    record.generation_speed,
    record.measurement_context,
    record.zhanh_note,
    record.felix_note,
    record.other_note,
    record.hanyu_note,
    record.author_org,
    record.domain,
    record.workstream,
    record.year,
    record.venue
  ]
    .join(" ")
    .toLocaleLowerCase();
}

function matchesFilters(record) {
  const filter = state.filters;
  const query = filter.search.trim().toLocaleLowerCase();
  if (query && !recordSearchText(record).includes(query)) return false;
  if (filter.domain && record.domain !== filter.domain) return false;
  if (state.excludedDomains.has(record.domain)) return false;
  if (filter.year && record.year !== filter.year) return false;
  if (filter.venue && record.venue !== filter.venue) return false;
  if (filter.workstream && record.workstream !== filter.workstream) return false;
  if (filter.zhanh && record.zhanh_rating !== filter.zhanh) return false;
  if (filter.other === "unrated" && record.other_rating) return false;
  if (filter.other && filter.other !== "unrated" && record.other_rating !== filter.other) {
    return false;
  }
  if (filter.organization && record.author_org !== filter.organization) return false;
  return true;
}

function compareRecords(left, right) {
  const key = state.sort;
  let result;
  if (key === "model_params_b") {
    if (!left[key] || !right[key]) {
      if (!left[key] && !right[key]) return left.source_order - right.source_order;
      return left[key] ? -1 : 1;
    }
    result = Number(left[key]) - Number(right[key]);
  } else if (key === "other_rating" || key === "zhanh_rating") {
    result = ratingValue(left[key]) - ratingValue(right[key]);
  } else if (key === "year" || key === "source_order") {
    result = Number(left[key]) - Number(right[key]);
  } else {
    result = collator.compare(left[key] ?? "", right[key] ?? "");
  }
  if (result === 0) result = left.source_order - right.source_order;
  return state.direction === "asc" ? result : -result;
}

function makeTag(value, kind) {
  const classSuffix = value.toLowerCase().replace(/[^a-z0-9]+/g, "-");
  const className =
    kind === "domain"
      ? `tag tag-domain tag-domain-${classSuffix}`
      : `tag tag-${classSuffix}`;
  return el("span", className, value);
}

function makeRating(value) {
  const provisional = value?.endsWith("*");
  const rating = el(
    "span",
    `rating${value ? "" : " rating-unrated"}${provisional ? " rating-provisional" : ""}`,
    value || "—"
  );
  if (provisional) rating.title = "Provisional rating pending reread";
  return rating;
}

function makeProjectNotes(record, compact = false) {
  const notes = [
    ["Zhanh", record.zhanh_note, "zhanh"],
    ["Felix", record.felix_note, "felix"],
    ["Other", record.other_note, "other"],
    ["Hanyu", record.hanyu_note, "hanyu"]
  ].filter(([, note]) => Boolean(note));

  if (notes.length === 0) {
    return [el("span", "note-empty", "No project note recorded")];
  }

  return notes.map(([name, note, author]) => {
    const paragraph = el(
      "p",
      `author-note author-${author}${compact ? " author-note-compact" : ""}`
    );
    paragraph.append(
      el("strong", "author-label", `${name}:`),
      document.createTextNode(` ${note}`)
    );
    return paragraph;
  });
}

function resourceDefinition(record) {
  return [
    ["paper_url", "Paper", "file-text"],
    ["github_url", record.github_note ? "Code*" : "Code", "github"],
    ["demo_url", "Demo", "play"],
    ["web_url", "Web", "globe-2"]
  ].filter(([key]) => Boolean(record[key]));
}

function makeResourceLink(record, definition) {
  const [key, label, icon] = definition;
  const link = el("a", "resource-link");
  link.href = record[key];
  link.target = "_blank";
  link.rel = "noreferrer";
  link.title = `Open ${label}`;
  const iconNode = document.createElement("i");
  iconNode.dataset.lucide = icon;
  iconNode.setAttribute("aria-hidden", "true");
  link.append(iconNode, document.createTextNode(label));
  return link;
}

function makeOrganization(record) {
  const node = el(record.author_org_url ? "a" : "span", "organization-name", record.author_org || "待核验");
  if (record.author_org_url) {
    node.href = record.author_org_url;
    node.target = "_blank";
    node.rel = "noreferrer";
    node.title = "First author's affiliation source";
  }
  return node;
}

function makeRow(record) {
  const row = document.createElement("tr");
  row.dataset.id = record.id;

  const yearCell = el("td", "year-venue");
  yearCell.append(el("strong", "", record.year || "待核验"), el("span", "", record.venue));

  const otherCell = el("td", "rating-column other-column");
  otherCell.append(makeRating(record.other_rating));
  const zhanhCell = el("td", "rating-column zhanh-column");
  zhanhCell.append(makeRating(record.zhanh_rating));

  const domainCell = el("td", "domain-column");
  domainCell.append(makeTag(record.domain, "domain"));
  const workstreamCell = el("td", "workstream-column");
  workstreamCell.append(makeTag(record.workstream, "workstream"));

  const titleCell = el("td", "title-column");
  const titleButton = el("button", "title-button", record.title);
  titleButton.type = "button";
  titleButton.addEventListener("click", () => openDetails(record));
  titleCell.append(titleButton);

  const keywordsCell = el("td", "keywords optional-wide", record.keywords);
  const metricsCell = el("td", "metrics-column optional-wide");
  metricsCell.append(
    el("p", "metric-size", record.model_size || (record.domain.startsWith("MusicGen") ? "规模待核验" : "")),
    el("p", "metric-speed", record.generation_speed || (record.domain.startsWith("MusicGen") ? "速度未核验" : ""))
  );
  metricsCell.title = record.measurement_context || "";
  const linksCell = el("td", "links-column");
  const linkList = el("div", "link-list");
  for (const definition of resourceDefinition(record)) {
    linkList.append(makeResourceLink(record, definition));
  }
  linksCell.append(linkList);
  const valueCell = el("td", "direct-value optional-wide");
  const noteStack = el("div", "note-stack note-stack-compact");
  noteStack.append(...makeProjectNotes(record, true));
  valueCell.append(noteStack);
  const organizationCell = el("td", "organization-column");
  organizationCell.append(makeOrganization(record));

  row.append(
    yearCell,
    otherCell,
    zhanhCell,
    domainCell,
    workstreamCell,
    titleCell,
    keywordsCell,
    metricsCell,
    linksCell,
    valueCell,
    organizationCell
  );
  return row;
}

function renderActiveFilters() {
  elements.activeFilters.replaceChildren();
  const labels = {
    search: "Search",
    domain: "Domain",
    year: "Year",
    venue: "Venue",
    workstream: "Workstream",
    zhanh: "Zhanh",
    other: "Other",
    organization: "主要作者"
  };
  for (const [key, value] of Object.entries(state.filters)) {
    if (value) elements.activeFilters.append(el("span", "filter-chip", `${labels[key]}: ${value}`));
  }
  for (const domain of [...state.excludedDomains].sort(collator.compare)) {
    const chip = el("button", "filter-chip filter-chip-removable");
    chip.type = "button";
    chip.title = `Restore ${domain}`;
    chip.setAttribute("aria-label", `Restore ${domain}`);
    chip.append(document.createTextNode(`Excluded: ${domain}`));
    const icon = document.createElement("i");
    icon.dataset.lucide = "x";
    icon.setAttribute("aria-hidden", "true");
    chip.append(icon);
    chip.addEventListener("click", () => {
      state.excludedDomains.delete(domain);
      syncControls();
      render();
    });
    elements.activeFilters.append(chip);
  }
}

function updateSortHeaders() {
  for (const button of document.querySelectorAll(".sort-button")) {
    const active = button.dataset.sort === state.sort;
    button.dataset.active = String(active);
    button.closest("th").setAttribute(
      "aria-sort",
      active ? (state.direction === "asc" ? "ascending" : "descending") : "none"
    );
    const icon = button.querySelector("i, svg");
    if (icon) {
      icon.setAttribute(
        "data-lucide",
        active ? (state.direction === "asc" ? "arrow-up" : "arrow-down") : "chevrons-up-down"
      );
    }
  }
}

function render() {
  syncControls();
  state.filtered = state.catalog.filter(matchesFilters).sort(compareRecords);
  elements.body.replaceChildren(...state.filtered.map(makeRow));
  elements.tableState.hidden = state.filtered.length > 0;
  if (state.filtered.length === 0) {
    elements.tableState.replaceChildren(
      document.createTextNode("No literature matches the current filters")
    );
  }
  elements.resultCount.textContent = `Showing ${state.filtered.length} of ${state.catalog.length} records`;
  renderActiveFilters();
  updateSortHeaders();
  updateUrl();
  refreshIcons();
}

function openDetails(record) {
  elements.detailMeta.textContent = `${record.year || "待核验"} · ${record.venue}`;
  elements.detailTitle.textContent = record.title;
  elements.detailLabels.replaceChildren(
    makeTag(record.domain, "domain"),
    makeTag(record.workstream, "workstream"),
    makeRating(record.zhanh_rating),
    makeRating(record.other_rating)
  );
  elements.detailKeywords.textContent = record.keywords || "No terms recorded";
  elements.detailNotes.replaceChildren(...makeProjectNotes(record));
  elements.detailMetrics.replaceChildren();
  for (const [label, value, source] of [
    ["参数量", record.model_size, record.model_size_url],
    ["生成速度", record.generation_speed, record.generation_speed_url]
  ]) {
    const paragraph = el("p", "", `${label}: ${value || (record.domain.startsWith("MusicGen") ? "未核验 / 未报告" : "未登记 / 不适用")}`);
    if (source) {
      const link = el("a", "metric-source", "source");
      link.href = source;
      link.target = "_blank";
      link.rel = "noreferrer";
      paragraph.append(document.createTextNode(" "), link);
    }
    elements.detailMetrics.append(paragraph);
  }
  if (record.measurement_context) elements.detailMetrics.append(el("p", "metric-context", record.measurement_context));
  elements.detailOrganization.replaceChildren(makeOrganization(record));
  elements.detailLinks.replaceChildren(
    ...resourceDefinition(record).map((definition) => makeResourceLink(record, definition))
  );
  elements.dialog.showModal();
  refreshIcons();
}

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.dataset.visible = "true";
  window.clearTimeout(showToast.timeout);
  showToast.timeout = window.setTimeout(() => {
    elements.toast.dataset.visible = "false";
  }, 2400);
}

function csvEscape(value) {
  const text = String(value ?? "");
  return `"${text.replaceAll('"', '""')}"`;
}

function exportFilteredCsv() {
  const rows = [CSV_HEADERS, ...state.filtered.map((record) => CSV_HEADERS.map((key) => record[key]))];
  const csv = rows.map((row) => row.map(csvEscape).join(",")).join("\n");
  const blob = new Blob([`\ufeff${csv}`], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "huawei-rl-literature-filtered.csv";
  anchor.click();
  URL.revokeObjectURL(url);
  showToast(`Exported ${state.filtered.length} records`);
}

function resetFilters() {
  for (const key of Object.keys(state.filters)) state.filters[key] = "";
  state.sort = "source_order";
  state.direction = "asc";
  state.excludedDomains.clear();
  syncControls();
  render();
}

function bindEvents() {
  const selects = {
    year: elements.year,
    venue: elements.venue,
    zhanh: elements.zhanh,
    other: elements.other,
    organization: elements.organization
  };
  elements.domain.addEventListener("change", (event) => {
    const checkbox = event.target.closest("input[data-domain]");
    if (!checkbox) return;
    if (checkbox.checked) state.excludedDomains.delete(checkbox.dataset.domain);
    else state.excludedDomains.add(checkbox.dataset.domain);
    render();
  });
  document.querySelector("#domains-all").addEventListener("click", () => {
    state.excludedDomains.clear();
    render();
  });
  document.querySelector("#domains-none").addEventListener("click", () => {
    state.excludedDomains = new Set(optionValues("domain"));
    render();
  });
  document.addEventListener("click", (event) => {
    if (!event.composedPath().includes(elements.domainMenu)) elements.domainMenu.open = false;
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") elements.domainMenu.open = false;
  });
  elements.search.addEventListener("input", () => {
    state.filters.search = elements.search.value;
    render();
  });
  for (const [key, select] of Object.entries(selects)) {
    select.addEventListener("change", () => {
      state.filters[key] = select.value;
      render();
    });
  }
  elements.workstream.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-value]");
    if (!button) return;
    state.filters.workstream = button.dataset.value;
    syncControls();
    render();
  });
  for (const button of document.querySelectorAll(".sort-button")) {
    button.addEventListener("click", () => {
      if (state.sort === button.dataset.sort) {
        state.direction = state.direction === "asc" ? "desc" : "asc";
      } else {
        state.sort = button.dataset.sort;
        state.direction = button.dataset.sort.includes("rating") ? "desc" : "asc";
      }
      render();
    });
  }
  elements.clear.addEventListener("click", resetFilters);
  elements.export.addEventListener("click", exportFilteredCsv);
  elements.copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(window.location.href);
      showToast("Filtered view link copied");
    } catch {
      showToast("Could not copy the link; copy it from the address bar");
    }
  });
  elements.dialogClose.addEventListener("click", () => elements.dialog.close());
  elements.dialog.addEventListener("click", (event) => {
    if (event.target === elements.dialog) elements.dialog.close();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "/" && !["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) {
      event.preventDefault();
      elements.search.focus();
    }
  });
}

async function init() {
  refreshIcons();
  window.addEventListener("load", refreshIcons, { once: true });
  const initialPaperId = new URLSearchParams(window.location.search).get("paper");
  readUrlState();
  bindEvents();
  try {
    const response = await fetch(DATA_URL, { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.catalog = await response.json();
    hydrateFilters();
    syncControls();
    elements.syncStatus.textContent = `${state.catalog.length} records · synced from GitHub`;
    render();

    const selected =
      initialPaperId && state.catalog.find((record) => record.id === initialPaperId);
    if (selected) openDetails(selected);
  } catch (error) {
    elements.syncStatus.textContent = "Catalog unavailable";
    elements.tableState.replaceChildren(
      document.createTextNode(`Could not load the catalog: ${error.message}`)
    );
    console.error(error);
  }
}

init();
