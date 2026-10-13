import assert from 'node:assert/strict';
import fs from 'node:fs/promises';

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const browser = await chromium.launch({ channel: process.env.PLAYWRIGHT_CHANNEL || 'chrome', headless: true });
const url = process.argv[2] || 'http://127.0.0.1:4179/demos/#lyrics';
const output = process.argv[3] || '/tmp/lyrics-objective-qa';
await fs.mkdir(output, { recursive: true });
try {
  for (const [name, width, height] of [['desktop', 1440, 1000], ['mobile', 390, 844]]) {
    const page = await browser.newPage({ viewport: { width, height } });
    const errors = [];
    page.on('pageerror', (e) => errors.push(e.message));
    await page.goto(url, { waitUntil: 'networkidle' });
    const root = page.locator('#lyrics-objective');
    await root.locator('[data-objective="body"] tr').first().waitFor();
    assert.equal(await root.locator('[data-objective="body"] tr').count(), 14);
    assert.equal(await root.locator('[data-objective="head"] th').count(), 11);
    assert.match(await root.locator('[data-objective="title"]').innerText(), /本站冻结 checkpoint/);
    const report = await page.evaluate(async () => (await fetch('./data/lyrics-objective-results.json')).json());
    const diagnosticCount = report.datasets.wsb.rows.reduce((n, r) => n + r.diagnostic_rows.length, 0);
    assert.equal(await root.locator('[data-objective="diagnostic-body"] tr').count(), diagnosticCount);
    if (diagnosticCount) {
      await root.locator('[data-objective="diagnostics"] summary').click();
      assert.equal(await root.locator('[data-objective="diagnostic-head"] th').count(), 11);
      assert.match(await root.locator('[data-objective="diagnostic-body"]').innerText(), /YuE2-Baseline/);
      await root.locator('[data-objective="diagnostics"] summary').click();
    }
    await root.locator('[data-objective-origin="official"]').click();
    assert.equal(await root.locator('[data-objective="body"] tr').count(), 17);
    assert.match(await root.locator('[data-objective="title"]').innerText(), /官方发表，非本站运行/);
    const yue = root.locator('[data-objective="body"] tr').filter({ has: page.getByRole('cell', { name: 'YuE2', exact: true }) });
    assert.match(await yue.innerText(), /6\.7316/);
    assert.match(await yue.innerText(), /8\.44%/);
    assert.equal(await root.locator('th').evaluateAll((cells) => cells.some((c) => c.scrollWidth > c.clientWidth + 1)), false, 'Header overflow');
    await root.locator('[data-objective-dataset="cmi"]').click();
    assert.equal(await root.locator('[data-objective="body"] tr').count(), 14);
    assert.equal(await root.locator('td[data-status="not-applicable"]').count(), 28);
    assert.equal(await root.locator('[data-objective-origin="official"]').isDisabled(), true);
    assert.match(await root.locator('[data-objective="title"]').innerText(), /CMI-Pref/);
    await root.locator('[data-objective-dataset="wsb"]').click();
    assert.equal(await root.locator('td[data-status="resource-pending"]').count(), 14);
    assert.equal(await root.locator('td[data-status="not-applicable"]').count(), 0);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1);
    assert.equal(overflow, false, 'Horizontal page overflow');
    await root.screenshot({ path: `${output}/${name}-local.png` });
    await root.locator('[data-objective-origin="official"]').click();
    await root.screenshot({ path: `${output}/${name}-published.png` });
    assert.deepEqual(errors, []);
    await page.close();
    console.log(`${name}: two datasets, 9 metrics, 17 published / 14 local settings, no page errors`);
  }
} finally { await browser.close(); }
