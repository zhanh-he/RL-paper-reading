import assert from 'node:assert/strict';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const {chromium} = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const site = fileURLToPath(new URL('../site/', import.meta.url));
const server = http.createServer((req, res) => {
  const filename = path.resolve(site, '.' + new URL(req.url, 'http://localhost').pathname);
  if (!filename.startsWith(site)) {res.writeHead(403).end(); return;}
  try {
    const target = fs.statSync(filename).isDirectory() ? path.join(filename, 'index.html') : filename;
    res.writeHead(200, {'Content-Type':'text/html; charset=utf-8'});
    fs.createReadStream(target).pipe(res);
  } catch {res.writeHead(404).end();}
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
let browser;
try {
  // Uses only Playwright's isolated managed Chromium, never an installed personal browser.
  browser = await chromium.launch({headless:true});
  const page = await browser.newPage();
  const output = process.env.QA_OUTPUT || '/tmp/amt-research-qa';
  fs.mkdirSync(output, {recursive:true});
  for (const width of [1440,390]) {
    await page.setViewportSize({width,height:960});
    for (const name of ['index','audio2score','generation','midi2score']) {
      await page.goto(`http://127.0.0.1:${server.address().port}/amt/research/2026-10-29/${name}.html`);
      assert.equal(await page.locator('h1').count(), 1);
      assert.ok((await page.locator('body').innerText()).includes('2026-10-29'));
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
      assert.equal(await page.locator('nav a').count(), 6);
      assert.equal(await page.getByRole('link', {name:'MusicGen / RL catalog',exact:true}).getAttribute('href'), '../../../');
      if (name === 'index') assert.equal(await page.locator('tbody tr').count(), 43);
      await page.screenshot({path:path.join(output, `${name}-${width}.png`), fullPage:true});
    }
    await page.getByRole('link', {name:'专题总览',exact:true}).click();
    assert.equal(await page.locator('tbody tr').count(), 43);
  }
  console.log('Four research pages: desktop/mobile rendering, table containment, navigation and 43 stable references verified.');
} finally {
  if (browser) await browser.close();
  server.closeAllConnections();
  await new Promise(resolve => server.close(resolve));
}
