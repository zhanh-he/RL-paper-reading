import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const CSV_PATH = path.join(ROOT, "data", "literature.csv");
const JSON_PATH = path.join(ROOT, "site", "data", "literature.json");
const README_PATH = path.join(ROOT, "README.md");
const CHECK_ONLY = process.argv.includes("--check");

const EXPECTED_HEADERS = [
  "id",
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
  "github_note",
  "demo_url",
  "web_url",
  "direct_value",
  "obsidian_target"
];

const DOMAINS = new Set([
  "MusicGen",
  "MusicEval",
  "MIR",
  "AudioGen",
  "SpeechEnhance",
  "AudioLLM",
  "LLM",
  "CV",
  "ML",
  "SourceSep",
  "Multimodal",
  "Other"
]);
const WORKSTREAMS = new Set(["Reward", "RL", "Reward-n-RL", "Other"]);
const RATING_PATTERN = /^(?:[0-5](?:\.5)?)\/5$/;

function parseCsv(input) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;

  for (let index = 0; index < input.length; index += 1) {
    const char = input[index];

    if (quoted) {
      if (char === '"' && input[index + 1] === '"') {
        field += '"';
        index += 1;
      } else if (char === '"') {
        quoted = false;
      } else {
        field += char;
      }
      continue;
    }

    if (char === '"' && field.length === 0) {
      quoted = true;
    } else if (char === ",") {
      row.push(field);
      field = "";
    } else if (char === "\n") {
      row.push(field.replace(/\r$/, ""));
      rows.push(row);
      row = [];
      field = "";
    } else {
      field += char;
    }
  }

  if (quoted) {
    throw new Error("CSV ended inside a quoted field");
  }
  if (field.length > 0 || row.length > 0) {
    row.push(field.replace(/\r$/, ""));
    rows.push(row);
  }
  return rows.filter((item) => item.some((value) => value !== ""));
}

function slugify(value) {
  return value
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-zA-Z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .toLowerCase()
    .slice(0, 96);
}

function validateUrl(value, rowNumber, field, errors) {
  if (!value) return;
  try {
    const url = new URL(value);
    if (url.protocol !== "https:" && url.protocol !== "http:") {
      errors.push(`row ${rowNumber}: ${field} must use http(s)`);
    }
  } catch {
    errors.push(`row ${rowNumber}: ${field} is not a valid URL`);
  }
}

function loadCatalog() {
  const parsed = parseCsv(fs.readFileSync(CSV_PATH, "utf8"));
  if (parsed.length < 2) throw new Error("The catalog CSV has no data rows");

  const [headers, ...records] = parsed;
  if (headers.join("\0") !== EXPECTED_HEADERS.join("\0")) {
    throw new Error(
      `Unexpected CSV headers. Expected: ${EXPECTED_HEADERS.join(", ")}`
    );
  }

  const errors = [];
  const ids = new Set();
  const titles = new Set();
  const catalog = records.map((values, index) => {
    const rowNumber = index + 2;
    if (values.length !== headers.length) {
      errors.push(
        `row ${rowNumber}: expected ${headers.length} columns, got ${values.length}`
      );
    }

    const record = Object.fromEntries(
      headers.map((header, column) => [header, values[column] ?? ""])
    );
    record.id = record.id || `${record.year}-${slugify(record.title)}`;
    record.source_order = index;
    record.keywords_list = record.keywords
      .split(/[；;]/)
      .map((item) => item.trim())
      .filter(Boolean);

    for (const field of ["year", "venue", "domain", "workstream", "title"]) {
      if (!record[field]) errors.push(`row ${rowNumber}: ${field} is required`);
    }
    if (!/^\d{4}$/.test(record.year)) {
      errors.push(`row ${rowNumber}: year must contain four digits`);
    }
    if (!DOMAINS.has(record.domain)) {
      errors.push(`row ${rowNumber}: unsupported domain ${record.domain}`);
    }
    if (!WORKSTREAMS.has(record.workstream)) {
      errors.push(`row ${rowNumber}: unsupported workstream ${record.workstream}`);
    }
    for (const field of ["felix_rating", "zhanh_rating"]) {
      if (record[field] && !RATING_PATTERN.test(record[field])) {
        errors.push(`row ${rowNumber}: invalid ${field} ${record[field]}`);
      }
    }
    if (record.keywords_list.length > 3) {
      errors.push(`row ${rowNumber}: keywords must contain at most three terms`);
    }
    if (ids.has(record.id)) errors.push(`row ${rowNumber}: duplicate id ${record.id}`);
    if (titles.has(record.title)) {
      errors.push(`row ${rowNumber}: duplicate title ${record.title}`);
    }
    ids.add(record.id);
    titles.add(record.title);

    for (const field of ["paper_url", "github_url", "demo_url", "web_url"]) {
      validateUrl(record[field], rowNumber, field, errors);
    }
    return record;
  });

  if (errors.length > 0) {
    throw new Error(`Catalog validation failed:\n- ${errors.join("\n- ")}`);
  }
  return catalog;
}

function escapeMarkdownCell(value) {
  return String(value ?? "")
    .replace(/\r?\n/g, "<br>")
    .replace(/\|/g, "\\|");
}

function markdownLinks(record) {
  const links = [];
  if (record.paper_url) links.push(`[paper](${record.paper_url})`);
  if (record.github_url) {
    const suffix = record.github_note ? ` (${record.github_note})` : "";
    links.push(`[github${suffix}](${record.github_url})`);
  }
  if (record.demo_url) links.push(`[demo](${record.demo_url})`);
  if (record.web_url) links.push(`[web](${record.web_url})`);
  return links.join(" · ");
}

function buildMarkdownTable(catalog) {
  const header = [
    "| 年份 / Venue | Felix 星级 | Zhanh 星级 | 主题 | 文献全名 | 核心词汇 | 超链接 | 对项目的直接价值 |",
    "|---|:---:|:---:|---|---|---|---|---|"
  ];
  const rows = catalog.map((record) => {
    const cells = [
      `${record.year}<br>${record.venue}`,
      record.felix_rating,
      record.zhanh_rating,
      `\`${record.domain}\` · \`${record.workstream}\``,
      record.title,
      record.keywords,
      markdownLinks(record),
      record.direct_value
    ].map(escapeMarkdownCell);
    return `| ${cells.join(" | ")} |`;
  });
  return [...header, ...rows].join("\n");
}

function updateReadme(catalog) {
  const start = "<!-- catalog:start -->";
  const end = "<!-- catalog:end -->";
  const current = fs.readFileSync(README_PATH, "utf8");
  const startIndex = current.indexOf(start);
  const endIndex = current.indexOf(end);
  if (startIndex < 0 || endIndex < 0 || endIndex <= startIndex) {
    throw new Error("README catalog markers are missing or out of order");
  }
  const generated = `${start}\n${buildMarkdownTable(catalog)}\n${end}`;
  return `${current.slice(0, startIndex)}${generated}${current.slice(
    endIndex + end.length
  )}`;
}

function writeOrCheck(filePath, next) {
  const current = fs.existsSync(filePath) ? fs.readFileSync(filePath, "utf8") : "";
  if (CHECK_ONLY) {
    if (current !== next) {
      throw new Error(`${path.relative(ROOT, filePath)} is out of date; run npm run build`);
    }
    return;
  }
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, next);
}

const catalog = loadCatalog();
const publicCatalog = catalog.map(({ obsidian_target, ...record }) => record);
const json = `${JSON.stringify(publicCatalog, null, 2)}\n`;
writeOrCheck(JSON_PATH, json);
writeOrCheck(README_PATH, updateReadme(catalog));

console.log(
  `${CHECK_ONLY ? "Checked" : "Built"} ${catalog.length} literature records successfully.`
);
