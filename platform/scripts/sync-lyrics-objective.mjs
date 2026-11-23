import fs from 'node:fs';
import { createHash } from 'node:crypto';
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
const pair = local.datasets.wsb.verified_muse_pair;
if (pair) {
  if (Object.keys(pair).sort().join(',') !== 'date,receipt,rl_id,sha256,stock_id' ||
      !/^\d{4}-\d{2}-\d{2}$/.test(pair.date) || pair.receipt !== 'wildsongbench-muse-paired.json' ||
      pair.stock_id !== 'muse-baseline' || pair.rl_id !== 'muse-mean7-v3-step270') throw new Error('Invalid Muse paired receipt metadata');
  const payload = fs.readFileSync(path.join(base, 'site/demos/data', pair.receipt));
  if (createHash('sha256').update(payload).digest('hex') !== pair.sha256) throw new Error('Muse paired receipt hash mismatch');
  const report = JSON.parse(payload);
  if (report.schema !== 'muse-wsb-paired-public-v1' || report.status !== 'complete' ||
      report.expected_prompts !== 192 || report.paired_prompts !== 192 ||
      report.selection !== 'best-per-of-two-four-asr-v1' ||
      Object.keys(report.metrics).sort().join(',') !== local.metrics.toSorted().join(',')) throw new Error('Incomplete Muse paired receipt');
  for (const [arm, id] of [['stock', pair.stock_id], ['rl', pair.rl_id]]) {
    const row = local.datasets.wsb.rows.find(r => r.id === id);
    if (!row || row.status !== 'complete' || row.generated_candidates !== 384 || row.expected_candidates !== 384 || row.scored_conditions !== 192 ||
        report[arm].candidate_audio_count !== 384 || report[arm].candidate_per_count !== 384) throw new Error('Incomplete Muse candidate coverage');
    if ((arm === 'stock' && row.step !== 0) || (arm === 'rl' && row.step !== 270)) throw new Error('Wrong Muse checkpoint step');
    if (row.receipt_files.length !== 1 || row.receipt_files[0].file !== 'platform/site/demos/data/wildsongbench-muse-paired.json' ||
        row.receipt_files[0].sha256 !== pair.sha256) throw new Error('Muse row is not bound to its paired receipt');
    for (const key of local.metrics) {
      const metric = row.metrics[key];
      if (report.metrics[key].n !== 192 || metric.status !== 'complete' || metric.scored !== 192 || metric.expected !== 192 ||
          !Number.isFinite(report.metrics[key].headline[arm]) || metric.value !== report.metrics[key].headline[arm]) throw new Error(`Muse paired metric mismatch: ${arm} ${key}`);
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
