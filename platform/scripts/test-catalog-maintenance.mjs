import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import { loadCatalog, parseCsv } from "./sync-catalog.mjs";
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
  assert(!table.includes("/articles/"));
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
