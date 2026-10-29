import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { CATALOGS, loadCatalog, parseCsv, renderCatalogPage } from "./sync-catalog.mjs";
import { catalogTable, replaceSection, sharedSection, updateFrontmatter } from "./sync-obsidian.mjs";

const records = loadCatalog();
const record = records[0];
const start = "<!-- shared-catalog:start -->";
const end = "<!-- shared-catalog:end -->";

test("CSV parsing retains quoted commas, multiline cells and escaped quotes", () => {
  assert.deepEqual(parseCsv('a,b\n"x,y","say ""yes""\nnext"\n'), [["a", "b"], ["x,y", 'say "yes"\nnext']]);
});

test("shared blocks preserve private prose and are idempotent", () => {
  const input = "---\ntitle: Example\nimportance: 4.5\n---\n\n# Example\n\nMy original research reasoning.\n";
  const block = sharedSection(record);
  const next = updateFrontmatter(replaceSection(input, block), record);
  assert(next.includes("My original research reasoning."));
  assert(next.includes("importance: 4.5"));
  assert.equal(updateFrontmatter(replaceSection(next, block), record), next);
});

test("malformed/duplicate markers fail rather than overwrite", () => {
  assert.throws(() => replaceSection(start + "bad", "next"), /Malformed/);
  assert.throws(() => replaceSection(start + end + start + end, "next"), /Duplicate/);
});

test("unchanged metadata can retain its sync receipt date", () => {
  const date = "2026-09-01";
  const input = "---\ntitle: Example\n---\n\n# Example\n";
  const next = updateFrontmatter(replaceSection(input, sharedSection(record, date)), record, date);
  assert(next.includes('catalog_synced: "2026-09-01"'));
  assert(next.includes("> 2026-09-01 同步自 GitHub"));
  assert.equal(updateFrontmatter(replaceSection(next, sharedSection(record, date)), record, date), next);
});

test("current table contains exactly one linked row per stable ID", () => {
  const table = catalogTable(records);
  assert.equal(table.split("\n").length, records.length + 2);
  assert.equal(new Set(records.map((r) => r.obsidian_target)).size, records.length);
  // Only vault paths must stay flat; legitimate publisher URLs may contain /articles/.
  assert(!/\[\[[^\]]*\/articles\//.test(table));
  for (const r of records) assert(table.includes(`${r.obsidian_target}\\|`));
});

test("understanding foundations, not generators; no ungrounded quantities", () => {
  assert(records.filter((r) => r.domain === "MusicFoundation").every((r) => /MERT|MuQ:/.test(r.title)));
  for (const r of records) {
    if (r.model_size) assert(r.model_size_url);
    if (r.generation_speed) assert(r.generation_speed_url && r.measurement_context);
    if (r.model_params_b) assert(Number(r.model_params_b) > 0);
  }
});

test("published JSON cannot reveal vault routing", () => {
  const json = JSON.parse(fs.readFileSync(new URL("../site/data/literature.json", import.meta.url), "utf8"));
  assert.equal(json.length, records.length);
  assert(json.every((r) => !Object.hasOwn(r, "obsidian_target")));
});

test("split catalogs are disjoint and have explicit ownership", () => {
  const byId = new Map(records.map(record => [record.id, record]));
  assert.equal(byId.size, records.length);
  assert.equal(loadCatalog("rl").length + loadCatalog("amt").length, records.length);
  assert(loadCatalog("amt").every(record => record.domain === "MusicTranscription"));
  assert(loadCatalog("rl").every(record => record.domain !== "MusicTranscription"));
  assert.throws(() => loadCatalog("unknown"), /Unknown catalog/);
});

test("each page and public dataset uses its own catalog, assets and request path", () => {
  for (const [id, config] of Object.entries(CATALOGS)) {
    const html = renderCatalogPage(id);
    const target = new URL(`../site/${config.route}index.html`, import.meta.url);
    assert.equal(fs.readFileSync(target, "utf8"), html);
    assert(html.includes(`data-catalog="${id}"`));
    assert(html.includes(`${config.base}assets/app.js`));
    assert(html.includes(`main/upd_request/${id}`));
    assert(html.includes('aria-current="page"'));
    assert(!html.includes("{{"));
    const json = JSON.parse(fs.readFileSync(new URL(`../site/data/${id}-literature.json`, import.meta.url), "utf8"));
    assert.deepEqual(json.map(record => record.id), loadCatalog(id).map(record => record.id));
    assert(json.every(record => record.catalog === id && !Object.hasOwn(record, "obsidian_target")));
    for (const person of ["zhanh", "other", "hanyu"]) {
      const request = fs.readFileSync(new URL(`../../upd_request/${id}/Update_request_${person}.txt`, import.meta.url), "utf8");
      assert(request.startsWith(`Catalog: ${id.toUpperCase()}`));
    }
  }
  const amt = loadCatalog("amt")[0];
  assert(sharedSection(amt).includes(`/amt/?paper=${encodeURIComponent(amt.id)}`));
});

test("Obsidian sync scopes AMT independently and keeps private prose with two indexes", t => {
  const vault = fs.mkdtempSync(path.join(os.tmpdir(), "catalog-vault-test-"));
  t.after(() => fs.rmSync(vault, { recursive: true, force: true }));
  const project = path.join(vault, "30-projects/huawei-rl");
  const literature = path.join(project, "1 Literature");
  fs.mkdirSync(literature, { recursive: true });
  fs.writeFileSync(path.join(project, "AGENTS.md"), "Test fixture rules\n");
  const rlIndex = path.join(literature, "Huawei RL 文献目录.md");
  const original = `# RL index\n\nPrivate research prose stays.\n\n${start}\nold table\n${end}\n`;
  fs.writeFileSync(rlIndex, original);
  const run = (...args) => spawnSync(process.execPath, [fileURLToPath(new URL("./sync-obsidian.mjs", import.meta.url)), "--vault", vault, "--date", "2026-10-22", ...args], { encoding: "utf8" });
  let result = run("--catalog", "amt", "--write");
  assert.equal(result.status, 0, result.stderr);
  assert.equal(fs.readFileSync(rlIndex, "utf8"), original);
  const amtIndex = fs.readFileSync(path.join(literature, "AMT 文献目录.md"), "utf8");
  for (const record of loadCatalog("amt")) assert(amtIndex.includes(record.title));
  assert(!amtIndex.includes(loadCatalog("rl")[0].title));
  result = run("--write");
  assert.equal(result.status, 0, result.stderr);
  assert(fs.readFileSync(rlIndex, "utf8").includes("Private research prose stays."));
  assert(!fs.readFileSync(rlIndex, "utf8").includes(loadCatalog("amt")[0].title));
  result = run();
  assert.equal(result.status, 0, result.stderr || result.stdout);
  assert.equal(JSON.parse(result.stdout).changes, 0);
});
