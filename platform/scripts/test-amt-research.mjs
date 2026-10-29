import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {loadCatalog} from './sync-catalog.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const registry = JSON.parse(fs.readFileSync(path.join(root, 'data/research-register-2026-10-29.json')));
const records = loadCatalog();
test('all 43 research entries resolve to exactly one stable catalog owner', () => {
  assert.equal(registry.items.length, 43);
  assert.equal(new Set(registry.items.map(x => x.register_id)).size, 43);
  for (const item of registry.items) {
    const matches = records.filter(record => record.id === item.catalog_id);
    assert.equal(matches.length, 1, item.register_id);
    assert.equal(matches[0].catalog, item.catalog);
    assert.equal(matches[0].title, item.title);
    assert.ok(['T', 'A', 'M'].includes(item.review_depth));
  }
});
test('guide pages preserve readable links without private vault content', () => {
  const folder = path.join(root, 'site/amt/research/2026-10-29');
  for (const name of ['index', 'audio2score', 'generation', 'midi2score']) {
    const filename = path.join(folder, `${name}.html`);
    const content = fs.readFileSync(filename, 'utf8');
    assert.ok(!/\/Users\/|obsidian:\/\/|\[\[|组内协调清单|sources\/2026-10-29\/pdf/.test(content));
    for (const match of content.matchAll(/href="([^"]+)"/g)) {
      if (/^https:\/\//.test(match[1])) continue;
      const target = path.resolve(folder, match[1]);
      assert.ok(fs.existsSync(target), `${name}: ${match[1]}`);
    }
  }
});
