import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const output = process.env.QA_OUTPUT || '/tmp/reward-regime-qa';
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true, channel: 'chrome' });
try {
  const page = await browser.newPage(); const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  for (const width of [1440, 390]) {
    await page.setViewportSize({ width, height: 1000 });
    await page.goto(process.env.DEMO_URL || 'http://127.0.0.1:4179/demos/#reward-regimes');
    const root = page.locator('#view-reward-regimes');
    await root.locator('[data-regime="models"] button').first().waitFor();
    assert.equal(await root.isVisible(), true);
    assert.equal(await root.locator('[data-regime="models"] button').count(), 5);
    for (const button of await root.locator('[data-regime="models"] button').all()) {
      await button.click();
      assert.equal(await button.getAttribute('aria-pressed'), 'true');
      assert.equal(await root.locator('[data-regime="arms"] tr').count(), 3);
      assert.match(await root.locator('[data-regime="model-note"]').textContent(), /MOS 未测/);
    }
    const data = await page.evaluate(async () => (await fetch('./data/reward-regime-protocol.json')).json());
    assert.equal(data.training_authorized, false); assert.deepEqual(data.runs, []);
    assert.equal(await root.locator('audio,canvas').count(), 0);
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth+1));
    await root.locator('[data-regime="models"] button').first().click();
    await page.screenshot({ path: `${output}/${width}.png`, fullPage: true });
  }
  assert.deepEqual(errors, []);
  console.log('SP/AI proposal: model controls, no fabricated results, desktop/mobile/no-overflow passed');
} finally { await browser.close(); }
