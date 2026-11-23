import assert from "node:assert/strict";
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadCatalog, parseCsv } from "./sync-catalog.mjs";

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const site = fileURLToPath(new URL("../site/", import.meta.url));
const mime = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".json": "application/json" };
let server;
let browser;
try {
  let base = process.env.CATALOG_BASE_URL;
  if (!base) {
    server = http.createServer((req, res) => {
      let target = path.resolve(site, "." + decodeURIComponent(new URL(req.url, "http://localhost").pathname));
      if (target !== path.resolve(site) && !target.startsWith(site)) { res.writeHead(403).end(); return; }
      try {
        if (fs.statSync(target).isDirectory()) target = path.join(target, "index.html");
        res.writeHead(200, { "Content-Type": mime[path.extname(target)] || "application/octet-stream" });
        fs.createReadStream(target).pipe(res);
      } catch { res.writeHead(404).end(); }
    });
    await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
    base = `http://127.0.0.1:${server.address().port}/`;
  }
  // Playwright's dedicated testing browser; never the user's installed Chrome.
  browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ acceptDownloads: true });
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  const output = process.env.QA_OUTPUT || "/tmp/rl-amt-catalog-qa";
  fs.mkdirSync(output, { recursive: true });
  for (const width of [1440, 390]) {
    await page.setViewportSize({ width, height: 1000 });
    for (const id of ["rl", "amt"]) {
      const records = loadCatalog(id);
      await page.goto(new URL(id === "rl" ? "./" : "amt/", base).href);
      await page.locator("#sync-status").filter({ hasText: "synced from GitHub" }).waitFor();
      assert.equal(await page.locator("#catalog-body tr").count(), records.length);
      assert.deepEqual(
        (await page.locator("thead th").allTextContents()).map(text => text.trim()),
        ["Literature", "Domain / Workstream", "Authors / Year / Venue", "Other", "Zhanh", "Direct Project Value"]
      );
      assert.equal(await page.locator(".catalog-switch [aria-current=page]").count(), 1);
      assert.match(await page.locator(".header-actions .button-primary").getAttribute("href"), new RegExp(`/upd_request/${id}$`));
      assert.equal(await page.locator(".catalog-switch").getByRole("link", { name: "MusicGen Live", exact: true }).getAttribute("href"), id === "rl" ? "./musicgen/" : "../musicgen/");
      assert.equal(await page.locator(".catalog-switch").getByRole("link", { name: "Music-RL Live", exact: true }).getAttribute("href"), id === "rl" ? "./demos/" : "../demos/");
      await page.locator("#search-input").fill(records[0].title);
      assert.equal(await page.locator("#catalog-body tr").count(), 1);
      const downloadEvent = page.waitForEvent("download");
      await page.locator("#export-csv").click();
      const download = await downloadEvent;
      assert.equal(download.suggestedFilename(), `${id}-literature-filtered.csv`);
      const exported = parseCsv(fs.readFileSync(await download.path(), "utf8").replace(/^\uFEFF/, ""));
      assert.equal(exported.length, 2);
      assert(exported[1].includes(records[0].title));
      assert(exported[0].includes("direct_project_value"));
      assert(!exported[0].some(key => key === "model_size" || key === "generation_speed"));
      await page.reload();
      await page.locator("#sync-status").filter({ hasText: "synced from GitHub" }).waitFor();
      assert.equal(await page.locator("#catalog-body tr").count(), 1);
      await page.locator("#clear-filters").click();
      await page.locator("#domain-menu summary").click();
      await page.locator("#domains-none").click();
      assert.equal(await page.locator("#catalog-body tr").count(), 0);
      await page.locator("#domains-all").click();
      assert.equal(await page.locator("#catalog-body tr").count(), records.length);
      await page.locator("#domain-menu summary").click();
      await page.locator(".sort-button[data-sort=title]").click();
      assert.equal(await page.locator("th[aria-sort=ascending] .sort-button").getAttribute("data-sort"), "title");
      const firstRow = page.locator("#catalog-body tr").first();
      const firstId = await firstRow.getAttribute("data-id");
      const record = records.find(item => item.id === firstId);
      const title = firstRow.locator(".title-button");
      assert(Number(await title.evaluate(node => getComputedStyle(node).fontWeight)) >= 700);
      assert.equal(await firstRow.locator(".direct-value").textContent(), record.direct_project_value || "—");
      if (record.keywords) {
        const terms = firstRow.locator(".title-column > .keywords");
        assert.equal(await terms.textContent(), record.keywords);
        assert.equal(await terms.evaluate(node => getComputedStyle(node).fontWeight), "400");
        assert.equal(await terms.evaluate(node => getComputedStyle(node).color), "rgb(166, 58, 50)");
      }
      if (record.paper_url) {
        assert.equal(await firstRow.locator(".title-column").getByRole("link", { name: "Paper" }).getAttribute("href"), record.paper_url);
      }
      await title.click();
      assert.equal(await page.locator("#detail-dialog").isVisible(), true);
      assert.equal(await page.locator("#detail-value").textContent(), record.direct_project_value || "—");
      assert.equal(await page.locator("#detail-metrics").count(), 0);
      assert.equal(await page.locator("#detail-reviews[open]").count(), 0);
      await page.locator("#close-dialog").click();
      const box = await page.locator(".catalog-switch").boundingBox();
      assert(box.x >= 0 && box.x + box.width <= width + 1);
      await page.screenshot({ path: path.join(output, `${id}-${width}.png`) });
      await page.locator(`.catalog-switch a${id === "rl" ? ":nth-child(2)" : ":nth-child(1)"}`).click();
      await page.locator("#sync-status").filter({ hasText: "synced from GitHub" }).waitFor();
      assert.equal(await page.locator("#catalog-body tr").count(), loadCatalog(id === "rl" ? "amt" : "rl").length);
    }
  }
  await page.goto(new URL("./?sort=other_rating&dir=desc", base).href);
  await page.locator("#sync-status").filter({ hasText: "synced from GitHub" }).waitFor();
  assert.equal(await page.locator("th[aria-sort=descending] .sort-button").getAttribute("data-sort"), "other_rating");
  await page.goto(new URL("./?sort=model_params_b&dir=desc", base).href);
  await page.locator("#sync-status").filter({ hasText: "synced from GitHub" }).waitFor();
  assert.equal(new URL(page.url()).searchParams.has("sort"), false);
  const foundations = loadCatalog("rl").filter(record => record.workstream === "基础知识");
  if (foundations.length) {
    await page.locator('#workstream-filter button[data-value="基础知识"]').click();
    assert.equal(await page.locator("#catalog-body tr").count(), foundations.length);
    for (const record of foundations.filter(record => record.primary_author)) {
      const row = page.locator("#catalog-body tr").filter({ hasText: record.title });
      assert.equal(await row.locator(".primary-author").textContent(), record.primary_author);
    }
  }
  for (const [entry, destination] of [["rl", "amt"], ["amt", "rl"]]) {
    const record = loadCatalog(destination)[0];
    const url = new URL(entry === "rl" ? "./" : "amt/", base);
    url.searchParams.set("paper", record.id);
    await page.goto(url.href);
    await page.locator("#detail-dialog[open]").waitFor();
    assert.equal(await page.locator("body").getAttribute("data-catalog"), destination);
    assert.equal(await page.locator("#detail-title").textContent(), record.title);
    await page.locator("#close-dialog").click();
  }
  assert.deepEqual(errors, []);
  console.log("Both catalogs: desktop/mobile rendering, isolated filters, CSV exports, sorting, details, request links, catalog switching and legacy deep links verified.");
} finally {
  if (browser) await browser.close();
  if (server) {
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
  }
}
