import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const output = process.env.QA_OUTPUT || '/tmp/reward-regime-qa';
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true, channel: 'chrome',args:['--autoplay-policy=no-user-gesture-required'] });
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
    assert.match(await canvas.getAttribute('aria-description'), /Step 270/);
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
    assert.equal(await root.locator('[data-regime="plots"]').isVisible(),true);
    await root.locator('[data-regime="reward-controls"] button').nth(2).click();
    assert.equal(await root.locator('[data-regime="plots"]').isVisible(),false);
    await root.locator('[data-regime="reward-controls"] button').first().click();
    const media=root.locator('[data-regime="media"]');
    assert.equal(await media.locator('audio').count(),2);
    const baseline=await media.locator('audio').first().getAttribute('src');
    const baselineSha=await media.locator('audio').first().getAttribute('data-source-sha');
    for(const step of ['50 steps','100 steps','270 steps']) {
      await root.locator('[data-regime="step-controls"]').getByRole('button',{name:step,exact:true}).click();
      assert.equal(await media.locator('audio').first().getAttribute('src'),baseline);
      assert.equal(await media.locator('audio').first().getAttribute('data-source-sha'),baselineSha);
    }
    await page.waitForFunction(()=>Array.from(document.querySelectorAll('[data-regime="media"] audio')).every(a=>Number.isFinite(a.duration)&&a.duration>=30));
    for(const audio of await media.locator('audio').all()) {
      assert.match(await audio.getAttribute('src'),/\.mp3$/);
      await audio.evaluate(a=>a.play());await audio.evaluate(a=>a.pause());
    }
    await media.scrollIntoViewIfNeeded();
    await page.waitForFunction(()=>Array.from(document.querySelectorAll('[data-regime="media"] img')).every(i=>i.complete&&i.naturalWidth>0));
    await media.screenshot({path:`${output}/media-${width}.png`});
    await root.locator('[data-mode="bad"]').click();
    assert.equal(await root.locator('[data-regime="example-controls"] button').count(),2);
    assert.match(await root.locator('[data-regime="replay-note"]').textContent(),/短验证音频/);
    assert.equal(await media.locator('audio').count(),2);
    await root.locator('[data-regime="reward-controls"] button').nth(1).click();
    assert.equal(await root.locator('[data-regime="example-controls"] button').count(),1);
    await root.locator('[data-regime="reward-controls"] button').first().click();
    await root.locator('[data-mode="long"]').click();
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
    await page.evaluate(()=>window.scrollTo(0,0));
    await page.screenshot({ path: `${output}/${width}.png`, fullPage: true });
  }
  assert.deepEqual(errors, []);
  console.log('SP/AI: curves/milestones, fixed baseline across50/100/270, playable30s+MP3, visuals,2Richness+1Instrument candidates, desktop/mobile/no-overflow passed');
} finally { await browser.close(); }
