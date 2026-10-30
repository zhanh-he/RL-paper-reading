import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const input = path.join(root, 'data/amt/score-study.json');
const output = path.join(root, 'site/amt/demos/data/score-study.json');
const data = JSON.parse(fs.readFileSync(input, 'utf8'));
if (!/^\d{4}-\d{2}-\d{2}$/.test(data.updated_at)) throw new Error('Missing score-study date');
for (const field of ['models', 'experiments', 'datasets', 'rewards']) if (!Array.isArray(data[field])) throw new Error(`Missing score-study ${field}`);
for (const model of data.models) {
  for (const field of ['name', 'family', 'output', 'benchmark', 'metrics', 'release', 'priority', 'paper', 'code']) if (typeof model[field] !== 'string' || !model[field]) throw new Error(`Missing model ${field}`);
  for (const field of ['paper', 'code']) if (new URL(model[field]).protocol !== 'https:') throw new Error('Unsafe source link');
}
const next = JSON.stringify(data, null, 2) + '\n';
if (process.argv.includes('--check')) {
  if (!fs.existsSync(output) || fs.readFileSync(output, 'utf8') !== next) throw new Error('Score study is stale; run build');
} else { fs.mkdirSync(path.dirname(output), { recursive: true }); fs.writeFileSync(output, next); }
console.log(`AMT score study: ${data.models.length} model families, ${data.experiments.length} benchmark experiments`);
