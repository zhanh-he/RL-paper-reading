import assert from 'node:assert/strict';
import { readFile, mkdir } from 'node:fs/promises';

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const url = process.env.DEMO_URL || 'http://127.0.0.1:4179/demos/#lyrics';
const output = process.env.QA_OUTPUT || '/tmp/framework-decisions-qa';
await mkdir(output, { recursive: true });
const protocol = JSON.parse(await readFile(new URL('../../music-gen/reward-regime-study/protocol.json', import.meta.url)));
const evidence = [];
for (const [model, file] of [['HeartMuLa', 'heartmula-trl'], ['SongBloom', 'songbloom-flow'], ['DiffRhythm2', 'diffrhythm2-flow']]) {
  const receipt = JSON.parse(await readFile(new URL(`../../music-gen/reward-regime-study/framework-smoke/${file}-receipt.json`, import.meta.url)));
  evidence.push({ model, memory: receipt.peak_allocated_gib.toFixed(3), seconds: (receipt.elapsed_seconds ?? receipt.seconds).toFixed(3) });
}
const browser = await chromium.launch({ headless: true, channel: 'chrome' });
try {
  for (const width of [1440, 390]) {
    const context = await browser.newContext({ viewport: { width, height: 1000 } });
    const page = await context.newPage();
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto(url);
    const link = page.locator('[data-framework-decision-link]');
    assert.equal(await link.isVisible(), true);
    assert.equal(await link.getAttribute('target'), '_blank');
    assert.match(await link.getAttribute('rel'), /noopener/);
    assert.equal(await link.evaluate((element) => element.closest('details, .lab-view')), null);
    const popupReady = page.waitForEvent('popup');
    await link.click();
    const decision = await popupReady;
    decision.on('pageerror', (error) => errors.push(error.message));
    const failed = [];
    decision.on('response', (response) => { if (response.status() >= 400) failed.push(response.url()); });
    await decision.waitForLoadState('networkidle');
    assert.match(decision.url(), /\/demos\/frameworks\.html$/);
    assert.equal(await decision.locator('h1').textContent(), 'TRL / Miles / VeRL');
    assert.equal(await decision.locator('[data-policy="ar"]').getAttribute('data-framework'), protocol.framework_transition.ar_framework);
    assert.equal(await decision.locator('[data-policy="flow"]').getAttribute('data-framework'), protocol.framework_transition.flow_framework);
    assert.equal(await decision.locator('details, .lab-view, button').count(), 0);
    assert.match(await decision.locator('#ar').textContent(), /FSDP LoRA 尚未实现/);
    assert.match(await decision.locator('#ar').textContent(), /beta=0/);
    assert.match(await decision.locator('#flow').textContent(), /仍未验收/);
    for (const row of evidence) {
      const text = await decision.locator('#evidence tbody tr').filter({ hasText: row.model }).textContent();
      assert.ok(text.includes(row.memory), text);
      assert.ok(text.includes(row.seconds), text);
    }
    for (const fragment of ['routes', 'ar', 'flow', 'evidence', 'sources']) {
      await decision.locator(`.decision-nav a[href="#${fragment}"]`).click();
      assert.equal(new URL(decision.url()).hash, `#${fragment}`);
      assert.ok(await decision.locator(`#${fragment}`).isVisible());
    }
    await decision.locator('h1').scrollIntoViewIfNeeded();
    assert.ok(await decision.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
    assert.equal(await decision.locator('.decision-sources a').count(), 8);
    assert.deepEqual(errors, []);
    assert.deepEqual(failed, []);
    await decision.screenshot({ path: `${output}/frameworks-${width}.png`, fullPage: true });
    await page.screenshot({ path: `${output}/entry-${width}.png` });
    await context.close();
  }
  console.log('Framework decisions: visible non-folded link, new tab, standalone page, pinned decisions, receipt values, anchors, desktop/mobile/no-overflow passed.');
} finally {
  await browser.close();
}
