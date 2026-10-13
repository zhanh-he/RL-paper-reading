import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
import path from 'node:path';

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const url = process.env.DEMO_URL || 'http://127.0.0.1:4179/demos/';
const output = process.env.QA_OUTPUT || '/tmp/lyrics-reference-qa';
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true, ...(process.env.PLAYWRIGHT_CHANNEL ? { channel: process.env.PLAYWRIGHT_CHANNEL } : {}) });
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1050 } });
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto(`${url.split('#')[0]}#lyrics-models`);
  await page.waitForFunction(() => document.querySelectorAll('#lyrics-model-reference-body tr[data-reference-id]').length === 8);
  const models = page.locator('#view-models');
  assert.equal(await models.isVisible(), true);
  assert.equal(await models.locator('[data-reference-task="lyrics"]').getAttribute('aria-pressed'), 'true');
  for (const [family, count] of [['ar', 4], ['hybrid', 1], ['flow', 2], ['archive', 1], ['all', 8]]) {
    await page.locator(`[data-model-family="${family}"]`).click();
    assert.equal(await page.locator('#lyrics-model-reference-body tr').count(), count);
  }
  await models.locator('[data-reference-task="vocal"]').click();
  assert.equal(new URL(page.url()).hash, '#vocal-models');
  assert.equal(await models.locator('[data-reference-panel="vocal"]').isVisible(), true);
  assert.match(await models.locator('[data-reference-panel="vocal"]').textContent(), /LaDA-Band/);
  await models.locator('[data-open-view="vocal-rewards"]').click();
  const rewards = page.locator('#view-reward-catalog');
  assert.equal(await rewards.locator('[data-reference-panel="vocal"]').isVisible(), true);
  assert.match(await rewards.locator('[data-reference-panel="vocal"]').textContent(), /Beat-v2 F1/);
  await rewards.locator('[data-reference-task="lyrics"]').click();
  assert.equal(await page.locator('#lyrics-reward-reference-body tr').count(), 9);
  assert.match(await rewards.locator('[data-reference-panel="lyrics"]').textContent(), /critique then score/);
  await rewards.locator('[data-dataset-target="lyrics"]').click();
  assert.equal(await page.locator('[data-dataset-task="lyrics"]').getAttribute('aria-pressed'), 'true');
  await page.locator('#lyrics-data-survey summary').click();
  assert.equal(await page.locator('#lyrics-dataset-reference-body tr').count(), 8);
  assert.match(await page.locator('[data-dataset-panel="lyrics"]').textContent(), /270/);
  assert.match(await page.locator('[data-dataset-panel="lyrics"]').textContent(), /121/);
  const catalog = await page.evaluate(async () => (await fetch('./data/lyrics-reference.json')).json());
  assert.equal(catalog.models.length, 8);
  assert.equal(catalog.rewards.length, 9);
  assert.equal(catalog.datasets.length, 8);
  assert.ok(catalog.models.find((row) => row.id === 'songbloom').unavailable_sources.length);
  for (const width of [1440, 768, 390]) {
    await page.setViewportSize({ width, height: 950 });
    for (const route of ['lyrics-models', 'lyrics-rewards', 'datasets', 'vocal-models', 'vocal-rewards']) {
      await page.evaluate((route) => { location.hash = route; }, route);
      const id = route.endsWith('models') ? 'models' : route.endsWith('rewards') ? 'reward-catalog' : 'datasets';
      await page.locator(`#view-${id}`).waitFor({ state: 'visible' });
      await page.locator(`#view-${id}`).scrollIntoViewIfNeeded();
      const scroll = await page.evaluate(() => document.documentElement.scrollWidth);
      assert.ok(scroll <= width + 1, `${route} at ${width}: document overflow ${scroll}`);
      await page.screenshot({ path: path.join(output, `${route}-${width}.png`) });
    }
  }
  for (const route of ['vocal-models', 'vocal-rewards', 'lyrics-models', 'lyrics-rewards']) {
    await page.goto(`${url.split('#')[0]}#${route}`);
    const id = route.endsWith('models') ? 'models' : 'reward-catalog';
    await page.locator(`#view-${id}`).waitFor({ state: 'visible' });
    assert.equal(await page.locator(`#view-${id} [data-reference-task="${route.startsWith('vocal') ? 'vocal' : 'lyrics'}"]`).getAttribute('aria-pressed'), 'true');
  }
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ models: catalog.models.length, rewards: catalog.rewards.length, datasets: catalog.datasets.length, widths: [1440, 768, 390], pageErrors: errors, output }, null, 2));
} finally {
  await browser.close();
}
