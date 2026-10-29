import assert from 'node:assert/strict';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { amtFixture } from './test-fixtures/amt-experiments.mjs';

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const site = fileURLToPath(new URL('../site/', import.meta.url));
const mime = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.mp3': 'audio/mpeg' };
const server = http.createServer((req, res) => {
  let target = path.resolve(site, '.' + decodeURIComponent(new URL(req.url, 'http://localhost').pathname));
  if (!target.startsWith(site)) { res.writeHead(403).end(); return; }
  try {
    if (fs.statSync(target).isDirectory()) target = path.join(target, 'index.html');
    res.writeHead(200, { 'Content-Type': mime[path.extname(target)] || 'application/octet-stream' });
    fs.createReadStream(target).pipe(res);
  } catch { res.writeHead(404).end(); }
});
let browser;
try {
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const base = `http://127.0.0.1:${server.address().port}`;
  // Dedicated Playwright Chromium with an isolated temporary profile.
  browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  const errors = []; page.on('pageerror', error => errors.push(error.message));
  const out = process.env.QA_OUTPUT || '/tmp/amt-experiments-qa'; fs.mkdirSync(out, { recursive: true });
  for (const width of [1440, 390]) {
    await page.setViewportSize({ width, height: 1000 });
    await page.goto(`${base}/amt/demos/`);
    await page.getByRole('status').filter({ hasText: '实验待配置' }).waitFor();
    assert.equal(await page.locator('#media-grid article').count(), 4);
    assert.equal(await page.locator('#reward-chart').isVisible(), false);
    assert.equal(await page.locator('#experiment-select').isDisabled(), true);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    await page.screenshot({ path: path.join(out, `empty-${width}.png`), fullPage: true });
    for (const tab of ['datasets', 'models', 'rewards', 'experiments']) {
      await page.locator(`[data-tab=${tab}]`).click();
      assert.equal(await page.locator(`#panel-${tab}`).isVisible(), true);
    }
  }
  await page.goto(`${base}/amt/`);
  await page.locator('.experiment-link').click();
  assert.match(page.url(), /\/amt\/demos\//);
  await page.route('**/amt/demos/data/experiments.json', route => route.fulfill({ json: amtFixture() }));
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto(`${base}/amt/demos/?run=test-only#experiments`);
  await page.locator('#experiment-title').filter({ hasText: 'TEST FIXTURE' }).waitFor();
  await page.locator('#reward-value').filter({ hasText: '100 steps' }).waitFor();
  assert.match(await page.locator('#metrics-body').textContent(), /60\.0% \(\+10\.0 pp\)/);
  assert.equal(await page.locator('#media-grid canvas').count(), 3);
  await page.locator('#media-grid button.amt-preview').first().click();
  assert.equal(await page.locator('#roll-dialog').isVisible(), true);
  await page.keyboard.press('Escape');
  const chart = await page.locator('#reward-chart').boundingBox();
  await page.mouse.move(chart.x + 48 + (chart.width - 64) / 6, chart.y + 40);
  assert.match(await page.locator('#reward-value').textContent(), /50 steps · 0\.5000/);
  await page.mouse.move(0, 0);
  await page.locator('[data-step="0"]').click();
  assert.equal(await page.locator('#media-grid article').last().locator('canvas').count(), 1);
  await page.locator('[data-step="300"]').click();
  assert.equal(await page.locator('#media-grid article').last().locator('canvas').count(), 0);
  assert.match(await page.locator('#media-grid article').last().textContent(), /训练输出待提供/);
  await page.locator('[data-step="100"]').click();
  await page.locator('#columns').selectOption('2');
  assert.equal(await page.locator('#media-grid').evaluate(el => getComputedStyle(el).gridTemplateColumns.split(' ').length), 2);
  await page.screenshot({ path: path.join(out, 'fixture-desktop.png'), fullPage: true });
  await page.locator('#sample-select').selectOption('sample-b');
  assert.equal(await page.locator('#media-grid canvas').count(), 0);
  assert.deepEqual(errors, []);
  console.log('AMT live: empty desktop/mobile, routing, checkpoint/missing output, hover traces, MIDI zoom, sample selection, metrics and layout verified.');
} finally {
  if (browser) await browser.close();
  server.closeAllConnections(); await new Promise(resolve => server.close(resolve));
}
