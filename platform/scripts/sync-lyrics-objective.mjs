import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const base = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const source = path.resolve(base, '../music-gen/lyrics2song/evaluation');
const check = process.argv.includes('--check');
const official = JSON.parse(fs.readFileSync(path.join(source, 'official-wsb-results.json')));
const local = JSON.parse(fs.readFileSync(path.join(source, 'results.json')));
if (official.rows.length !== 17 || official.metrics.length !== 9 || official.rows.some((r) => r.n !== 192)) throw new Error('Official WSB snapshot mismatch');
if (local.schema !== 1 || local.datasets.wsb.conditions !== 192 || local.datasets.cmi.conditions !== 121) throw new Error('Local evaluation split mismatch');
for (const [dataset, data] of Object.entries(local.datasets)) {
  if (new Set(data.rows.map((r) => r.id)).size !== data.rows.length) throw new Error('Duplicate experiment rows');
  for (const r of data.rows) {
    if (r.expected_candidates !== data.conditions * 2 || r.generated_candidates > r.expected_candidates || r.scored_conditions > data.conditions) throw new Error('Invalid evaluation coverage');
    for (const m of official.metrics) {
      const c = r.metrics[m.key];
      const expected = data.conditions * (dataset === 'wsb' ? 1 : 2);
      if (!c || !['complete', 'pending', 'not-applicable', 'resource-pending', 'deferred'].includes(c.status)) throw new Error('Unknown metric state');
      if (!Number.isInteger(c.scored) || c.scored < 0 || c.scored > expected) throw new Error('Invalid metric coverage');
      if (c.status === 'complete' && (!Number.isFinite(c.value) || c.expected !== expected || c.scored !== expected)) throw new Error('Premature complete metric');
      if (c.status !== 'complete' && c.value !== null) throw new Error('Partial mean published as complete');
      if (dataset === 'cmi' && ['per', 'control_overall'].includes(m.key) && c.status !== 'not-applicable') throw new Error('Unsupported CMI metric must be n/a');
    }
  }
}
for (const [name, value] of [['wsb-official-results.json', official], ['lyrics-objective-results.json', local]]) {
  const target = path.join(base, 'site/demos/data', name);
  const rendered = JSON.stringify(value, null, 2) + '\n';
  if (check) { if (fs.readFileSync(target, 'utf8') !== rendered) throw new Error(`Outdated ${name}`); }
  else fs.writeFileSync(target, rendered);
}
console.log('Lyrics objective: 17 published settings, 9 metrics, CMI/WSB coverage validated');
