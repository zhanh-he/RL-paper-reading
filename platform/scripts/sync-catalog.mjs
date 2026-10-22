import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const JSON_PATH = path.join(ROOT, "site", "data", "literature.json");
const CHECK_ONLY = process.argv.includes("--check");

export const CATALOGS = {
  rl: { title: "MusicGen / RL / Reward", description: "Music generation, reinforcement learning and reward research", route: "./", base: "./" },
  amt: { title: "Music Transcription (AMT)", description: "Automatic music transcription, multipitch estimation and voice assignment", route: "amt/", base: "../" }
};

const EXPECTED_HEADERS = [
  "id",
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
  "github_note",
  "demo_url",
  "web_url",
  "zhanh_note",
  "felix_note",
  "other_note",
  "hanyu_note",
  "author_org",
  "author_org_url",
  "obsidian_target"
];

const DOMAINS = new Set([
  "MusicGen",
  "MusicGen(LLM)",
  "MusicGen(Small)",
  "MusicGen(Mid)",
  "MusicGen(Large)",
  "MusicFoundation",
  "MusicEval",
  "MusicTranscription",
  "MIR",
  "AudioGen",
  "Speech",
  "SpeechEnhance",
  "AudioLLM",
  "LLM",
  "CV",
  "MachineLearning",
  "SourceSep",
  "Multimodal",
  "Other"
]);
const WORKSTREAMS = new Set(["Reward", "RL", "Reward-n-RL", "Other"]);
const RATING_PATTERN = /^(?:[0-5](?:\.5)?)\/5\*?$/;

export function parseCsv(input) {
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

function loadSource(catalogId) {
  const filename = path.join(ROOT, "data", catalogId, "literature.csv");
  const parsed = parseCsv(fs.readFileSync(filename, "utf8"));
  if (parsed.length < 1) throw new Error(`${catalogId}: missing CSV headers`);

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
    record.catalog = catalogId;
    record.source_order = index;
    record.keywords_list = record.keywords
      .split(/[；;]/)
      .map((item) => item.trim())
      .filter(Boolean);

    for (const field of ["venue", "domain", "workstream", "title"]) {
      if (!record[field]) errors.push(`row ${rowNumber}: ${field} is required`);
    }
    if (!(record.year === "" && record.venue.startsWith("Web")) && !/^\d{4}$/.test(record.year)) {
      errors.push(`row ${rowNumber}: year must contain four digits`);
    }
    if (!DOMAINS.has(record.domain)) {
      errors.push(`row ${rowNumber}: unsupported domain ${record.domain}`);
    }
    if (!WORKSTREAMS.has(record.workstream)) {
      errors.push(`row ${rowNumber}: unsupported workstream ${record.workstream}`);
    }
    for (const field of ["other_rating", "zhanh_rating"]) {
      if (record[field] && !RATING_PATTERN.test(record[field])) {
        errors.push(`row ${rowNumber}: invalid ${field} ${record[field]}`);
      }
    }
    if (record.keywords_list.length > 3) {
      errors.push(`row ${rowNumber}: keywords must contain at most three terms`);
    }
    if (record.model_params_b && (!/^\d+(?:\.\d+)?$/.test(record.model_params_b) || Number(record.model_params_b) <= 0)) {
      errors.push(`row ${rowNumber}: model_params_b must be a positive number in billions`);
    }
    if (record.model_size && !record.model_size_url) {
      errors.push(`row ${rowNumber}: reported model size requires a primary source`);
    }
    if (record.generation_speed && (!record.generation_speed_url || !record.measurement_context)) {
      errors.push(`row ${rowNumber}: reported speed requires source and measurement context`);
    }
    if (ids.has(record.id)) errors.push(`row ${rowNumber}: duplicate id ${record.id}`);
    if (titles.has(record.title)) {
      errors.push(`row ${rowNumber}: duplicate title ${record.title}`);
    }
    ids.add(record.id);
    titles.add(record.title);

    for (const field of ["paper_url", "github_url", "demo_url", "web_url", "author_org_url", "model_size_url", "generation_speed_url"]) {
      validateUrl(record[field], rowNumber, field, errors);
    }
    return record;
  });

  if (errors.length > 0) {
    throw new Error(`Catalog validation failed:\n- ${errors.join("\n- ")}`);
  }
  return catalog;
}

export function loadCatalog(catalogId) {
  if (catalogId && !Object.hasOwn(CATALOGS, catalogId)) throw new Error(`Unknown catalog: ${catalogId}`);
  const records = Object.keys(CATALOGS).flatMap(loadSource);
  const ids = new Set();
  const titles = new Set();
  for (const record of records) {
    if (ids.has(record.id) || titles.has(record.title)) throw new Error(`Duplicate across catalogs: ${record.id}`);
    ids.add(record.id);
    titles.add(record.title);
  }
  return records.filter((record) => !catalogId || record.catalog === catalogId);
}

export function renderCatalogPage(catalogId) {
  const config = CATALOGS[catalogId];
  if (!config) throw new Error(`Unknown catalog: ${catalogId}`);
  const template = fs.readFileSync(path.join(ROOT, "templates", "catalog.html"), "utf8");
  const values = { CATALOG_ID: catalogId, TITLE: config.title, DESCRIPTION: config.description, BASE: config.base,
    RL_CURRENT: catalogId === "rl" ? 'aria-current="page"' : "",
    AMT_CURRENT: catalogId === "amt" ? 'aria-current="page"' : "" };
  return template.replace(/\{\{([A-Z_]+)\}\}/g, (_, key) => {
    if (!Object.hasOwn(values, key)) throw new Error(`Unknown template token: ${key}`);
    return values[key];
  });
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

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const catalog = loadCatalog();
  const publicCatalog = catalog.map(({ obsidian_target, ...record }) => record);
  const json = `${JSON.stringify(publicCatalog, null, 2)}\n`;
  // Preserve the old combined endpoint for existing integrations and paper links.
  writeOrCheck(JSON_PATH, json);
  for (const [catalogId, config] of Object.entries(CATALOGS)) {
    const records = publicCatalog.filter((record) => record.catalog === catalogId);
    writeOrCheck(path.join(ROOT, "site", "data", `${catalogId}-literature.json`), `${JSON.stringify(records, null, 2)}\n`);
    writeOrCheck(path.join(ROOT, "site", config.route, "index.html"), renderCatalogPage(catalogId));
    console.log(`${catalogId}: ${records.length} records`);
  }
  console.log(`${CHECK_ONLY ? "Checked" : "Built"} ${catalog.length} literature records successfully.`);
}
