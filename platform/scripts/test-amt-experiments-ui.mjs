import assert from 'node:assert/strict';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { amtFixture } from './test-fixtures/amt-experiments.mjs';

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const site = fileURLToPath(new URL('../site/', import.meta.url));
const mime = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.mp3': 'audio/mpeg', '.svg': 'image/svg+xml' };
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
  await page.route('**/amt/demos/data/experiments.json', route => route.fulfill({ json: {schema_version:1,updated_at:'2026-10-29',experiments:[],datasets:[],models:[],rewards:[]} }));
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
  await page.unroute('**/amt/demos/data/experiments.json');
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
  await page.unroute('**/amt/demos/data/experiments.json');
  for (const width of [1440,390]) {
    await page.setViewportSize({width,height:1000});
    await page.goto(`${base}/amt/demos/?run=metric-blindspots#experiments`);
    await page.locator('#metrics-body tr').filter({hasText:'整段高八度'}).waitFor();
    assert.equal(await page.locator('#training-section').isVisible(),false);
    assert.equal(await page.locator('#metrics-body tr').count(),5);
    assert.match(await page.locator('#metric-head').textContent(),/Pitch-class onset/);
    await page.locator('[data-key="all_notes_up_one_octave"]').click();
    assert.equal(await page.locator('#metrics-body tr[data-selected=true] td').nth(2).innerText(),'100.00%');
    assert.equal(await page.locator('#metrics-body tr[data-selected=true] td').nth(3).innerText(),'0.00%');
    assert.equal(await page.locator('#media-grid canvas').count(),3);
    assert.ok(await page.locator('#media-grid canvas').last().evaluate(canvas => {
      const pixels=canvas.getContext('2d').getImageData(0,0,canvas.width,canvas.height).data;
      let colored=0;for(let i=0;i<pixels.length;i+=4)if(pixels[i]>100&&pixels[i+1]<140&&pixels[i+2]<140)colored++;
      return colored>100;
    }));
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),true);
    await page.screenshot({path:path.join(out,`controls-${width}.png`),fullPage:true});
    await page.locator('#experiment-select').selectOption('f0-correlation-controls');
    assert.match(await page.locator('#metrics-body').textContent(),/1200\.00/);
    assert.equal(await page.locator('#media-grid').isVisible(),false);
    await page.locator('#experiment-select').selectOption('ying-midi-interface');
    assert.equal(await page.locator('#metrics-body tr').count(),3);
    assert.equal(await page.locator('#metrics-body tr').nth(1).locator('td').last().innerText(),'0.00%');
    assert.equal(await page.locator('#metrics-body tr').nth(2).locator('td').last().innerText(),'100.00%');
    await page.locator('#experiment-select').selectOption('sheetsage2-real-vocal-pilot');
    assert.equal(await page.locator('#metrics-body').getByText('0.00%',{exact:true}).count(),0);
    assert.equal(await page.locator('#media-grid canvas').count(),0);
    await page.locator('[data-tab=models]').click();
    assert.equal(await page.locator('#reported-results-body tr').count(),6);
    assert.match(await page.locator('#paper-comparison').innerText(),/非本地复现/);
    assert.match(await page.locator('#reported-results-body').innerText(),/83\.08/);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),true);
    await page.screenshot({path:path.join(out,`reported-models-${width}.png`),fullPage:true});
  }
  for (const width of [1440,390]) {
    await page.setViewportSize({width,height:1000});
    await page.goto(`${base}/amt/demos/?run=asap-duet-benchmark#experiments`);
    await page.locator('#score-results-body tr').nth(3).waitFor();
    assert.match(await page.locator('#score-results-body').innerText(), /86\.23% \/ 77\.98%/);
    assert.match(await page.locator('#score-results-body').innerText(), /40 updates/);
    assert.equal(await page.locator('#media-grid .amt-score-preview img').count(),4);
    await page.waitForFunction(() => [...document.querySelectorAll('#media-grid img')].every(image => image.complete && image.naturalWidth > 0));
    assert.equal(await page.locator('#media-grid a').count(),4);
    assert.match(await page.locator('#score-input').innerText(), /Real performance MIDI/);
    assert.equal(await page.locator('#metrics-body tr').count(),3);
    assert.equal(await page.locator('#score-paired-body tr').count(),8);
    assert.equal(await page.locator('#score-gtzan-body tr').count(),10);
    assert.equal(await page.locator('#score-grpo-trace tr').count(),50);
    await page.locator('#media-grid .amt-score-preview').last().click();
    assert.equal(await page.locator('#large-score').isVisible(),true);
    assert.equal(await page.locator('#large-roll').isVisible(),false);
    await page.keyboard.press('Escape');
    await page.locator('#sample-select').selectOption('test-07');
    await page.waitForFunction(() => [...document.querySelectorAll('#media-grid img')].every(image => image.complete && image.naturalWidth > 0 && image.src.includes('test-07')));
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),true);
    await page.screenshot({path:path.join(out,`real-score-${width}.png`),fullPage:true});
    await page.locator('[data-tab=models]').click();
    assert.equal(await page.locator('#score-model-body tr').count(),14);
    await page.locator('[data-score-family=midi]').click();
    assert.equal(await page.locator('#score-model-body tr').count(),3);
    assert.match(await page.locator('#score-model-body').innerText(), /MIDI2ScoreTransformer/);
    await page.locator('[data-score-family=all]').click();
    assert.equal(await page.locator('#score-model-body tr').count(),14);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),true);
    await page.locator('[data-tab=datasets]').click();
    assert.equal(await page.locator('#score-data-body tr').count(),5);
    await page.locator('[data-tab=rewards]').click();
    assert.equal(await page.locator('#score-rewards-body tr').count(),4);
  }
  assert.deepEqual(errors, []);
  console.log('AMT live: empty desktop/mobile, routing, checkpoint/missing output, hover traces, MIDI zoom, sample selection, metrics and layout verified.');
} finally {
  if (browser) await browser.close();
  server.closeAllConnections(); await new Promise(resolve => server.close(resolve));
}
