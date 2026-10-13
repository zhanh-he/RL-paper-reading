import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';

const source = new URL('../../music-gen/lyrics2song/research/catalog.json', import.meta.url);
const output = new URL('../site/demos/data/lyrics-reference.json', import.meta.url);
const data = JSON.parse(await readFile(source, 'utf8'));
assert.equal(data.schema_version, 1);
assert.match(data.checked_at, /^\d{4}-\d{2}-\d{2}$/);
const required = {
  models: ['architecture', 'family', 'input', 'release', 'license', 'posttrain'],
  rewards: ['role', 'definition', 'use', 'risk'],
  datasets: ['kind', 'content', 'protocol', 'boundary'],
};
for (const [group, fields] of Object.entries(required)) {
  assert.ok(data[group].length, `${group} cannot be empty`);
  const ids = new Set();
  for (const row of data[group]) {
    assert.match(row.id, /^[a-z0-9-]+$/);
    assert.ok(!ids.has(row.id), `Duplicate ${group} id: ${row.id}`);
    ids.add(row.id);
    for (const field of ['name', ...fields]) assert.ok(typeof row[field] === 'string' && row[field].trim(), `${row.id}: missing ${field}`);
    assert.ok(row.sources.length, `${row.id}: no primary source`);
    for (const link of row.sources) {
      assert.ok(link.label.trim());
      assert.equal(new URL(link.url).protocol, 'https:');
    }
    if (group === 'models') assert.ok(['ar', 'flow', 'hybrid', 'archive'].includes(row.family));
  }
}
const text = `${JSON.stringify(data, null, 2)}\n`;
if (process.argv.includes('--check')) {
  assert.equal(await readFile(output, 'utf8'), text, 'Lyrics reference data is stale; run the build');
} else {
  await writeFile(output, text);
}
console.log(`Lyrics reference verified: ${data.models.length} models, ${data.rewards.length} reward families, ${data.datasets.length} data resources (${data.checked_at})`);
