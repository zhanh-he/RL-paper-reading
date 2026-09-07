const DATA_URL = "./data/literature.json";
const CSV_HEADERS = [
  "year",
  "venue",
  "felix_rating",
  "zhanh_rating",
  "domain",
  "workstream",
  "title",
  "keywords",
  "paper_url",
  "github_url",
  "demo_url",
  "web_url",
  "zhanh_note",
  "felix_note"
];

const elements = {
  search: document.querySelector("#search-input"),
  domain: document.querySelector("#domain-filter"),
  year: document.querySelector("#year-filter"),
  venue: document.querySelector("#venue-filter"),
  zhanh: document.querySelector("#zhanh-filter"),
  felix: document.querySelector("#felix-filter"),
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
  detailLinks: document.querySelector("#detail-links"),
  toast: document.querySelector("#toast")
};

const state = {
  catalog: [],
  filtered: [],
  filters: {
    search: "",
    domain: "",
    year: "",
    venue: "",
    workstream: "",
    zhanh: "",
    felix: ""
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
  populateSelect(elements.domain, optionValues("domain"));
  populateSelect(elements.year, optionValues("year", true));
  populateSelect(elements.venue, optionValues("venue"));
  populateSelect(elements.zhanh, optionValues("zhanh_rating", true));
  populateSelect(elements.felix, optionValues("felix_rating", true));
}

function readUrlState() {
  const params = new URLSearchParams(window.location.search);
  for (const key of Object.keys(state.filters)) {
    state.filters[key] = params.get(key) ?? "";
  }
  state.sort = params.get("sort") || "source_order";
  state.direction = params.get("dir") === "desc" ? "desc" : "asc";
}

function syncControls() {
  elements.search.value = state.filters.search;
  elements.domain.value = state.filters.domain;
  elements.year.value = state.filters.year;
  elements.venue.value = state.filters.venue;
  elements.zhanh.value = state.filters.zhanh;
  elements.felix.value = state.filters.felix;
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
    record.zhanh_note,
    record.felix_note,
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
  if (filter.year && record.year !== filter.year) return false;
  if (filter.venue && record.venue !== filter.venue) return false;
  if (filter.workstream && record.workstream !== filter.workstream) return false;
  if (filter.zhanh && record.zhanh_rating !== filter.zhanh) return false;
  if (filter.felix === "unrated" && record.felix_rating) return false;
  if (filter.felix && filter.felix !== "unrated" && record.felix_rating !== filter.felix) {
    return false;
  }
  return true;
}

function compareRecords(left, right) {
  const key = state.sort;
  let result;
  if (key === "felix_rating" || key === "zhanh_rating") {
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
    ["Felix", record.felix_note, "felix"]
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

function makeRow(record) {
  const row = document.createElement("tr");
  row.dataset.id = record.id;

  const yearCell = el("td", "year-venue");
  yearCell.append(el("strong", "", record.year), el("span", "", record.venue));

  const felixCell = el("td", "rating-column felix-column");
  felixCell.append(makeRating(record.felix_rating));
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

  row.append(
    yearCell,
    felixCell,
    zhanhCell,
    domainCell,
    workstreamCell,
    titleCell,
    keywordsCell,
    linksCell,
    valueCell
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
    felix: "Felix"
  };
  for (const [key, value] of Object.entries(state.filters)) {
    if (value) elements.activeFilters.append(el("span", "filter-chip", `${labels[key]}: ${value}`));
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
  elements.detailMeta.textContent = `${record.year} · ${record.venue}`;
  elements.detailTitle.textContent = record.title;
  elements.detailLabels.replaceChildren(
    makeTag(record.domain, "domain"),
    makeTag(record.workstream, "workstream"),
    makeRating(record.zhanh_rating),
    makeRating(record.felix_rating)
  );
  elements.detailKeywords.textContent = record.keywords || "No terms recorded";
  elements.detailNotes.replaceChildren(...makeProjectNotes(record));
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
  syncControls();
  render();
}

function bindEvents() {
  const selects = {
    domain: elements.domain,
    year: elements.year,
    venue: elements.venue,
    zhanh: elements.zhanh,
    felix: elements.felix
  };
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
