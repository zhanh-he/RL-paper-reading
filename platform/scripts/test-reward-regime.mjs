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
    assert.equal(await root.locator('[data-regime="models"] button').count(), 6);
    for (const button of await root.locator('[data-regime="models"] button').all()) {
      await button.click();
      assert.equal(await button.getAttribute('aria-pressed'), 'true');
      assert.equal(await root.locator('[data-regime="arms"] tr').count(), 3);
      assert.match(await root.locator('[data-regime="model-note"]').textContent(), /MOS 未测/);
    }
    const data = await page.evaluate(async () => (await fetch('./data/reward-regime-protocol.json')).json());
    assert.equal(data.training_authorized, true); assert.equal(data.runs.length, 6);
    assert.deepEqual(data.design.lr_grid,['3e-4']);
    assert.equal(data.rewards[2].name, 'SongBench-Arrangement');
    assert.equal(await root.locator('audio').count(), 0);
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth+1));
    await root.locator('[data-regime="models"] button').first().click();
    const canvas=root.locator('[data-regime="reward-chart"]');
    await canvas.waitFor({state:'visible'});
    assert.match(await canvas.getAttribute('aria-description'), /Step 50/);
    assert.match(await root.locator('[data-regime="arms"]').textContent(), /30\/30/);
    const pixels=await canvas.evaluate((c)=>{
      const values=c.getContext('2d').getImageData(0,0,c.width,c.height).data;
      return values.filter((v,i)=>i%4===3 && v>0).length;
    });assert.ok(pixels>500);
    await canvas.scrollIntoViewIfNeeded();
    const bounds=await canvas.boundingBox();
    await page.mouse.move(bounds.x+39+(bounds.width-51)*50/270,bounds.y+70);
    await root.locator('.training-chart-tooltip:not([hidden])').first().waitFor();
    assert.match(await root.locator('.training-chart-tooltip:not([hidden])').first().textContent(),/Step 50/);
    await root.locator('[data-regime="reward-controls"] button').nth(1).click();
    assert.equal(await root.locator('[data-regime="plots"]').isVisible(),false);
    await root.locator('[data-regime="reward-controls"] button').first().click();
    await page.evaluate(()=>window.scrollTo(0,0));
    await page.screenshot({ path: `${output}/${width}.png`, fullPage: true });
  }
  assert.deepEqual(errors, []);
  console.log('SP/AI formal: approved matrix, measured curves, hover milestones, full-validation labels, desktop/mobile/no-overflow passed');
} finally { await browser.close(); }
