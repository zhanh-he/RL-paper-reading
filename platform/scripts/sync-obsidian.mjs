import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import { loadCatalog } from "./sync-catalog.mjs";

const dateIndex = process.argv.indexOf("--date");
const DATE = dateIndex >= 0 ? process.argv[dateIndex + 1] : new Intl.DateTimeFormat("en-CA", {
  timeZone: "Australia/Perth", year: "numeric", month: "2-digit", day: "2-digit"
}).format(new Date());
if (!/^\d{4}-\d{2}-\d{2}$/.test(DATE)) throw new Error("Invalid --date; expected YYYY-MM-DD");
const PROJECT = "30-projects/huawei-rl";
const START = "<!-- shared-catalog:start -->";
const END = "<!-- shared-catalog:end -->";
const INLINE_FIELDS = ["id", "year", "venue", "domain", "workstream", "other_rating", "zhanh_rating"];

export function replaceSection(current, next) {
  const start = current.indexOf(START);
  const end = current.indexOf(END);
  if (current.split(START).length > 2 || current.split(END).length > 2) {
    throw new Error("Duplicate shared markers; refusing ambiguous overwrite");
  }
  if ((start >= 0) !== (end >= 0) || (start >= 0 && end < start)) {
    throw new Error("Malformed shared catalog markers; refusing to overwrite note");
  }
  if (start >= 0) return current.slice(0, start) + next + current.slice(end + END.length);
  const heading = /^# .+$/m.exec(current);
  if (!heading) return `${current.trimEnd()}\n\n${next}\n`;
  const position = heading.index + heading[0].length;
  return current.slice(0, position) + `\n\n${next}` + current.slice(position);
}

export function updateFrontmatter(current, record, date = DATE) {
  if (!current.startsWith("---\n")) throw new Error(`Missing YAML frontmatter: ${record.id}`);
  const end = current.indexOf("\n---", 4);
  if (end < 0) throw new Error(`Malformed YAML frontmatter: ${record.id}`);
  const lines = current.slice(4, end).split("\n").filter((line) => !/^catalog_[a-z_]+:/.test(line));
  for (const field of INLINE_FIELDS) lines.push(`catalog_${field}: ${JSON.stringify(record[field] ?? "")}`);
  lines.push(`catalog_synced: ${JSON.stringify(date)}`);
  return `---\n${lines.join("\n")}${current.slice(end)}`;
}

function cell(value) {
  return String(value ?? "").replaceAll("|", "&#124;").replaceAll("\n", "<br>");
}

function wiki(record) {
  return `[[${PROJECT}/1 Literature/${record.obsidian_target}\\|${cell(record.title)}]]`;
}

function links(record) {
  return [["paper", "paper_url"], ["github", "github_url"], ["demo", "demo_url"], ["web", "web_url"]]
    .filter(([, key]) => record[key])
    .map(([label, key]) => `[${label}${key === "github_url" && record.github_note ? " (unofficial)" : ""}](${record[key]})`)
    .join(" · ");
}

function judgments(record) {
  return [["Zhanh", "zhanh_note"], ["Felix（原记录）", "felix_note"], ["Other", "other_note"], ["Hanyu", "hanyu_note"]]
    .filter(([, key]) => record[key])
    .map(([name, key]) => `**${name}**: ${record[key]}`);
}

export function sharedSection(record, date = DATE) {
  const metrics = [
    `- 参数量：${record.model_size || "未核验 / 未报告 / 不适用"}${record.model_size_url ? `；[来源](${record.model_size_url})` : ""}`,
    `- 速度：${record.generation_speed || "未登记可比较的定量速度"}${record.generation_speed_url ? `；[来源](${record.generation_speed_url})` : ""}`,
    record.measurement_context ? `- 测量口径：${record.measurement_context}` : ""
  ].filter(Boolean);
  return [START, "## 当前共享目录信息", "",
    `> ${date} 同步自 GitHub catalog。该区块是当前共享元数据；下方原有正文及旧 frontmatter 保留为阅读历史，不能把旧评分、旧发布状态或旧‘当前’声明当成刚核验的结论。`, "",
    `- 完整标题：${record.title}`,
    `- 年份 / Venue：${record.year || "待核验"} / ${record.venue}`,
    `- 分类：\`${record.domain}\` · \`${record.workstream}\``,
    `- Zhanh：${record.zhanh_rating || "未评分"}；Other：${record.other_rating || "未评分"}（来源归属见下方）`,
    `- 核心词汇：${record.keywords || "未登记"}`,
    `- 主要作者机构：${record.author_org || "待核验"}${record.author_org_url ? `；[核验来源](${record.author_org_url})` : ""}`,
    `- 公开入口：${links(record)}`,
    `- [Online catalog 条目](https://zhanh-he.github.io/RL-paper-reading/?paper=${encodeURIComponent(record.id)})`, "",
    ...metrics, "", "### 共享项目价值与独立判断", "",
    ...(judgments(record).length ? judgments(record).flatMap((text) => [text, ""]) : ["尚未登记个人判断；未评分不代表不重要。", ""]),
    END].join("\n");
}

export function catalogTable(records) {
  return ["| 年份 / Venue | Other 星级 | Zhanh 星级 | 主题 | 文献全名 | 核心词汇 | 规模 / 速度 | 超链接 | 对项目的直接价值 | 主要作者 |",
    "|---|:---:|:---:|---|---|---|---|---|---|---|",
    ...records.map((r) => {
      const value = judgments(r).map(cell).join("<br><br>");
      const metrics = [r.model_size, r.generation_speed, r.measurement_context].filter(Boolean).map(cell).join("<br>");
      const organization = r.author_org_url ? `[${cell(r.author_org)}](${r.author_org_url})` : cell(r.author_org);
      return `| ${r.year || "待核验"}<br>${cell(r.venue)} | ${r.other_rating} | ${r.zhanh_rating} | \`${r.domain}\` · \`${r.workstream}\` | ${wiki(r)} | ${cell(r.keywords)} | ${metrics} | ${links(r)} | ${value} | ${organization} |`;
    })].join("\n");
}

function newCard(record, pack) {
  const type = /WildSongBench/.test(record.title) ? "dataset-note" : record.venue.startsWith("Web") ? "resource-note" : "paper-note";
  const intro = pack?.summary || record.zhanh_note || record.other_note || "已建立元数据与项目入口；具体方法和实验设置待原文精读，不以目录收录替代科学核验。";
  const lines = ["---", `title: ${JSON.stringify(record.title)}`, `type: ${type}`,
    'status: "catalog-synced-awaiting-deep-reading"', `created: ${JSON.stringify(DATE)}`, `updated: ${JSON.stringify(DATE)}`,
    "project: huawei-rl", "tags:", "  - huawei-rl/literature", "  - catalog/synced", `accessed: ${JSON.stringify(DATE)}`, "---", "",
    `# ${record.title}`, "", "## 中文简介", "", intro, ""];
  if (pack) {
    lines.push("## 推荐研究包阅读线索", "", "> 归属：Jiahao 推荐研究包（原 request 模板名为 Felix，以 Zhanh 的推荐人说明为准）。这些说明用于续读；本轮重点核验分类、参数、速度及特定争议，不宣称全部方法数字均已独立重测。", "");
    for (const [label, field] of [["架构与训练", "arch"], ["后训练", "post"], ["评测证据", "evidence"], ["限制与风险", "limit"], ["可参考的设计", "take"]]) {
      if (pack[field]) lines.push(`### ${label}`, "", pack[field], "");
    }
  }
  lines.push("## 本地心得与待办", "", "- 待精读原文并记录适用条件、失败边界与可复现实验。", "- 论文报告、推荐人判断、我们的实验结果分别记录，不把它们混为同一证据。", "",
    `相关：[[${PROJECT}/0 Project Brief/2026-10-14_Huawei_RL_Research_Synthesis_and_Reading_Priorities|当前研究总结]] · [[${PROJECT}/1 Literature/Huawei RL 文献目录|文献总目录]]`, "");
  return lines.join("\n");
}

function readPack(filename) {
  if (!filename) return new Map();
  const content = fs.readFileSync(filename, "utf8");
  const start = content.indexOf("[\n");
  if (start < 0) throw new Error("Reading request contains no JSON array");
  let quoted = false, escaped = false, depth = 0, end = -1;
  for (let i = start; i < content.length; i++) {
    const c = content[i];
    if (quoted) { if (escaped) escaped = false; else if (c === "\\") escaped = true; else if (c === '"') quoted = false; }
    else if (c === '"') quoted = true;
    else if (c === "[") depth++;
    else if (c === "]" && --depth === 0) { end = i + 1; break; }
  }
  const items = JSON.parse(content.slice(start, end));
  const provenance = JSON.parse(fs.readFileSync(new URL("../data/catalog-provenance-20261014.json", import.meta.url)));
  return new Map(provenance.items.map((p) => [p.catalog_id, items.find((item) => item.id === p.request_item)]));
}

function main() {
  const args = process.argv.slice(2);
  const option = (name) => { const i = args.indexOf(name); return i < 0 ? null : args[i + 1]; };
  const vault = option("--vault");
  if (!vault) throw new Error("Pass --vault /path/to/vault; default is read-only validation, --write enables scoped updates");
  const write = args.includes("--write");
  const projectRoot = path.join(path.resolve(vault), PROJECT);
  if (!fs.existsSync(path.join(projectRoot, "AGENTS.md"))) throw new Error("Project rules missing; refusing to create a competing vault");
  const records = loadCatalog();
  const pack = readPack(option("--reading-pack"));
  const changes = [];
  for (const record of records) {
    if (!record.obsidian_target || path.basename(record.obsidian_target) !== record.obsidian_target) throw new Error(`Unsafe or stale routing: ${record.id}`);
    const target = path.join(projectRoot, "1 Literature", record.obsidian_target + ".md");
    const current = fs.existsSync(target) ? fs.readFileSync(target, "utf8") : "";
    const body = current || newCard(record, pack.get(record.id));
    const priorDate = /^catalog_synced: "(\d{4}-\d{2}-\d{2})"$/m.exec(current)?.[1] || DATE;
    const unchangedCandidate = updateFrontmatter(replaceSection(body, sharedSection(record, priorDate)), record, priorDate);
    const next = current === unchangedCandidate ? current : updateFrontmatter(replaceSection(body, sharedSection(record)), record);
    if (current !== next) {
      changes.push({ id: record.id, note: path.relative(projectRoot, target), action: current ? "updated-shared-block" : "created-reading-entry" });
      if (write) fs.writeFileSync(target, next);
    }
  }
  const target = path.join(projectRoot, "1 Literature/Huawei RL 文献目录.md");
  const current = fs.readFileSync(target, "utf8");
  const table = `${START}\n${catalogTable(records)}\n${END}`;
  if (!current.includes(START)) throw new Error("Catalog requires prepared shared markers; preserve its previous full snapshot before conversion");
  const next = replaceSection(current, table);
  if (next !== current) { changes.push({ note: path.relative(projectRoot, target), action: "updated-shared-table" }); if (write) fs.writeFileSync(target, next); }
  const report = { synced_at: DATE, source: "platform/data/literature.csv", records: records.length, mode: write ? "write" : "check", changes };
  if (option("--report")) fs.writeFileSync(option("--report"), JSON.stringify(report, null, 2) + "\n");
  console.log(JSON.stringify({ records: records.length, mode: report.mode, changes: changes.length, newCards: changes.filter((x) => x.action === "created-reading-entry").length }, null, 2));
  if (!write && changes.length) process.exitCode = 1;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
