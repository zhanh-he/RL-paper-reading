import assert from 'node:assert/strict';
import { readFile, mkdir } from 'node:fs/promises';
import path from 'node:path';

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const url = process.env.DEMO_URL || 'http://127.0.0.1:4179/demos/#lyrics';
const output = process.env.QA_OUTPUT || '/tmp/formal-replay-qa';
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true, ...(process.env.PLAYWRIGHT_CHANNEL ? { channel: process.env.PLAYWRIGHT_CHANNEL } : {}) });
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  if (process.env.LOCAL_RELEASE_ASSETS) {
    await page.route('https://github.com/zhanh-he/RL-paper-reading/releases/download/formal-cmi-pref-20261012/*', async (route) => {
      const file = path.join(process.env.LOCAL_RELEASE_ASSETS, path.basename(new URL(route.request().url()).pathname));
      const body = await readFile(file);
      await route.fulfill({ status: 200, contentType: 'audio/flac', body });
    });
  }
  await page.goto(url);
  await page.locator('#formal-v2-controls button').first().waitFor();
  const registry = await page.evaluate(async () => (await fetch('./formal-9to1-runs.json')).json());
  const root = page.locator('#lyrics-formal');
  let pairs = 0;
  for (const arm of registry.arms) {
    await root.locator(`#formal-v2-controls button[data-id="${arm.id}"]`).click();
    for (const example of registry.examples) {
      await root.locator(`#formal-v2-examples button[data-index="${example.index}"]`).click();
      const expectedBaseline = arm.long_replays['0']?.find((r) => r.index === example.index);
      if (expectedBaseline) assert.equal(await root.locator('[data-role="baseline"]').getAttribute('data-sha256'), expectedBaseline.sha256);
      for (const [step, clips] of Object.entries(arm.long_replays)) {
        const expected = clips.find((r) => r.index === example.index);
        if (!expected) continue;
        await root.locator('#formal-v2-stages button').filter({ hasText: new RegExp(`^${step} steps$`) }).click();
        assert.equal(await root.locator('[data-role="candidate"]').getAttribute('data-sha256'), expected.sha256);
        assert.equal(await root.locator('[data-role="baseline"]').getAttribute('data-sha256'), expectedBaseline.sha256);
        await root.locator('[data-role="candidate"] img').first().scrollIntoViewIfNeeded();
        await page.waitForFunction(() => [...document.querySelectorAll('#formal-v2-compare img')].every((i) => i.complete && i.naturalWidth > 0));
        pairs++;
      }
    }
  }
  await root.locator('#formal-v2-controls button').first().click();
  await root.locator('#formal-v2-examples button').first().click();
  const target = root.locator('[data-role="candidate"] audio');
  await target.evaluate((audio) => audio.load());
  await page.waitForFunction(() => document.querySelector('#formal-v2-compare [data-role="candidate"] audio').readyState >= 1, { timeout: 45000 });
  const media = await target.evaluate(async (audio) => { audio.currentTime = 30; await audio.play(); return { duration: audio.duration, paused: audio.paused, src: audio.currentSrc }; });
  assert.ok(media.duration >= 30); assert.equal(media.paused, false);
  await target.evaluate((audio) => audio.pause());
  await root.locator('#formal-v2-controls').scrollIntoViewIfNeeded();
  await page.screenshot({ path: path.join(output, 'desktop.png'), fullPage: false });
  for (const width of [390, 768]) {
    await page.setViewportSize({ width, height: 900 });
    await root.locator('#formal-v2-controls').scrollIntoViewIfNeeded();
    await page.screenshot({ path: path.join(output, `mobile-${width}.png`) });
    const overflow = await page.evaluate(() => ({ width: innerWidth, scroll: document.documentElement.scrollWidth,
      elements: [...document.querySelectorAll('body *')].filter((e) => { const r = e.getBoundingClientRect(); return r.width > 0 && r.right > innerWidth + 1 && getComputedStyle(e).position !== 'absolute'; }).slice(0, 20).map((e) => ({ tag: e.tagName, id: e.id, class: e.className, right: e.getBoundingClientRect().right })) }));
    assert.ok(overflow.scroll <= width + 1, `Horizontal overflow: ${JSON.stringify(overflow)}`);
  }
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ arms: registry.arms.length, replayPairsChecked: pairs, media, pageErrors: errors, output }, null, 2));
} finally { await browser.close(); }
