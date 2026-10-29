import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const root = resolve(import.meta.dirname, '..');
const sourcePath = resolve(root, 'data/musicgen/experiments.json');
const output = resolve(root, 'site/musicgen/data');
export const sha256 = (value) => createHash('sha256').update(value).digest('hex');
const canonical = (value) => `${JSON.stringify(value, null, 2)}\n`;
const tasks = ['vocal2accomp', 'lyrics2song', 'shared', 'extension'];
const safeId = /^[a-z0-9][a-z0-9._-]{0,95}$/;
const date = /^\d{4}-\d{2}-\d{2}$/;
const publicGitHubProjects = new Set(['zhanh-he/RL-paper-reading', 'yuhui1038/Muse', 'multimodal-art-projection/YuE', 'ace-step/ACE-Step-1.5']);
function fields(object, keys, label) {
  assert(object && typeof object === 'object' && !Array.isArray(object), `${label}: expected object`);
  assert.deepEqual(Object.keys(object).sort(), [...keys].sort(), `${label}: missing or unexpected public fields`);
}
function text(value, label) {
  assert.equal(typeof value, 'string', `${label}: expected text`);
  assert(value.trim().length > 0 && value.length <= 4000, `${label}: empty or oversized`);
}
function id(value) { assert(typeof value === 'string' && safeId.test(value), 'Expected anonymous public ID'); }
function choice(value, options) { assert(options.includes(value), `Unexpected value: ${value}`); }
function count(value) { assert(Number.isInteger(value) && value > 0, 'Expected positive integer count'); }
function rows(value, label, check) {
  assert(Array.isArray(value), `${label}: expected array`);
  const ids = new Set();
  for (const row of value) {
    check(row);
    id(row.id);
    assert(!ids.has(row.id), `${label}: duplicate ID`);
    ids.add(row.id);
  }
}
function texts(values) { assert(Array.isArray(values) && values.length > 0); values.forEach(value => text(value, 'list item')); }
function publicStrings(value) {
  if (Array.isArray(value)) return value.forEach(publicStrings);
  if (value && typeof value === 'object') return Object.values(value).forEach(publicStrings);
  if (typeof value !== 'string') return;
  assert(!/(?:\/Users\/|\/home\/|\/cache\/|\/mnt\/|\/tmp\/|\/workspace\/|file:\/\/|obs:\/\/|s3:\/\/|[A-Za-z]:\\)/i.test(value), 'Private/local path rejected');
  assert(!/(?:gh[pousr]_[A-Za-z0-9]{15,}|hf_[A-Za-z0-9]{20,}|-----BEGIN.*PRIVATE KEY)/i.test(value), 'Credential rejected');
  for (const match of value.matchAll(/https?:\/\/github\.com\/([^/\s]+)\/([^/#?\s]+)/gi)) {
    assert(publicGitHubProjects.has(`${match[1]}/${match[2].replace(/\.git$/, '')}`), 'Unreviewed repository link rejected');
  }
  assert(!/[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}/.test(value), 'Personal contact identifier rejected');
  assert(!/[\u0000-\u0008\u000b\u000c\u000e-\u001f]/.test(value), 'Control character rejected');
}

export function validateMusicGen(data) {
  fields(data, ['schema_version', 'updated_at', 'project', 'hypotheses', 'capabilities', 'ablations', 'protocol', 'roadmap', 'evidence', 'training_results', 'result_policy', 'references'], 'dataset');
  assert.equal(data.schema_version, 1);
  assert(date.test(data.updated_at));
  fields(data.project, ['title', 'subtitle', 'objective', 'primary_task', 'secondary_task', 'approach', 'boundary', 'snapshot_note'], 'project');
  Object.entries(data.project).forEach(([key, value]) => text(value, key));
  choice(data.project.primary_task, tasks.slice(0, 2));
  choice(data.project.secondary_task, tasks.slice(0, 2));
  rows(data.hypotheses, 'hypotheses', row => {
    fields(row, ['id', 'title', 'question', 'test', 'status'], 'hypothesis');
    ['title', 'question', 'test'].forEach(key => text(row[key], key));
    assert.equal(row.status, 'proposed');
  });
  rows(data.capabilities, 'capabilities', row => {
    fields(row, ['id', 'title', 'task', 'priority', 'status', 'input', 'output', 'boundary'], 'capability');
    ['title', 'priority', 'input', 'output', 'boundary'].forEach(key => text(row[key], key));
    choice(row.task, tasks);
    choice(row.status, ['planned', 'implemented_unverified']);
  });
  rows(data.ablations, 'ablations', row => {
    fields(row, ['id', 'label', 'score_plan', 'nar', 'status', 'question'], 'ablation');
    assert.equal(typeof row.score_plan, 'boolean');
    assert.equal(typeof row.nar, 'boolean');
    assert.equal(row.status, 'planned');
    text(row.label, 'label'); text(row.question, 'question');
  });
  assert.equal(data.ablations.length, 4, 'The preregistration proposal must contain all four factorial cells');
  assert.equal(new Set(data.ablations.map(row => `${row.score_plan}:${row.nar}`)).size, 4, 'Duplicate factorial cell');
  fields(data.protocol, ['status', 'registration_note', 'controls', 'evaluation', 'benchmark_note'], 'protocol');
  assert.equal(data.protocol.status, 'proposed');
  text(data.protocol.registration_note, 'registration_note'); text(data.protocol.benchmark_note, 'benchmark_note');
  texts(data.protocol.controls); texts(data.protocol.evaluation);
  rows(data.roadmap, 'roadmap', row => {
    fields(row, ['id', 'title', 'status', 'gate'], 'roadmap');
    text(row.title, 'title'); text(row.gate, 'gate'); choice(row.status, ['planned', 'in_progress']);
  });
  rows(data.evidence, 'evidence', row => {
    fields(row, ['id', 'date', 'kind', 'status', 'task', 'title', 'scope', 'tests_passed_per_suite', 'suite_count', 'cpu_update_steps', 'finding', 'limits', 'provenance'], 'engineering evidence');
    assert(date.test(row.date)); assert.equal(row.kind, 'engineering_check'); assert.equal(row.status, 'verified');
    choice(row.task, tasks);
    ['title', 'scope', 'finding', 'limits', 'provenance'].forEach(key => text(row[key], key));
    ['tests_passed_per_suite', 'suite_count', 'cpu_update_steps'].forEach(key => count(row[key]));
  });
  rows(data.training_results, 'training_results', row => {
    fields(row, ['id', 'experiment_id', 'date', 'task', 'evidence_kind', 'status', 'config_id', 'split_id', 'seed', 'metrics', 'summary'], 'training result');
    [row.experiment_id, row.config_id, row.split_id].forEach(id);
    assert(date.test(row.date)); choice(row.task, tasks.slice(0, 2));
    choice(row.evidence_kind, ['training_measurement', 'quality_measurement']);
    choice(row.status, ['completed', 'failed']);
    assert(Number.isInteger(row.seed) && row.seed >= 0); text(row.summary, 'summary');
    rows(row.metrics, 'metrics', metric => {
      fields(metric, ['id', 'value', 'n', 'unit', 'direction', 'scope'], 'metric');
      assert(metric.value === null || (typeof metric.value === 'number' && Number.isFinite(metric.value)), 'Metric must be finite or null');
      if (metric.n !== null) count(metric.n);
      if (metric.value !== null) count(metric.n);
      text(metric.unit, 'unit'); choice(metric.direction, ['higher_is_better', 'lower_is_better']);
      choice(metric.scope, ['training_diagnostic', 'quality', 'resource']);
      if (row.evidence_kind === 'training_measurement') assert.notEqual(metric.scope, 'quality', 'Training diagnostics cannot claim music quality');
    });
    if (row.status === 'failed') assert.equal(row.metrics.length, 0, 'Failed runs cannot claim evaluation scores');
    if (row.evidence_kind === 'quality_measurement' && row.status === 'completed') assert(row.metrics.some(metric => metric.scope === 'quality' && metric.value !== null), 'Quality claim needs a measured quality metric');
  });
  fields(data.result_policy, ['empty_message', 'workflow', 'retained_fields', 'excluded_fields'], 'result_policy');
  Object.entries(data.result_policy).forEach(([key, value]) => text(value, key));
  rows(data.references, 'references', row => {
    fields(row, ['id', 'title', 'url', 'relation'], 'reference');
    text(row.title, 'title'); text(row.relation, 'relation');
    const url = new URL(row.url); assert.equal(url.protocol, 'https:'); assert(!url.username && !url.password);
  });
  publicStrings(data);
  return data;
}

export async function syncMusicGen(checkOnly = false) {
  const source = await readFile(sourcePath, 'utf8');
  const data = validateMusicGen(JSON.parse(source));
  const dataset = canonical(data);
  const snapshot = canonical({ schema_version: 1, updated_at: data.updated_at, source_sha256: sha256(source), dataset_sha256: sha256(dataset) });
  if (!checkOnly) await mkdir(output, { recursive: true });
  for (const [name, content] of [['experiments.json', dataset], ['snapshot.json', snapshot]]) {
    const path = resolve(output, name);
    if (checkOnly) assert.equal(await readFile(path, 'utf8'), content, `Stale MusicGen output: ${name}`);
    else await writeFile(path, content);
  }
  console.log(`MusicGen snapshot ${checkOnly ? 'checked' : 'built'}: ${data.capabilities.length} task plans, ${data.evidence.length} engineering summaries, ${data.training_results.length} training/quality results.`);
}
if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  await syncMusicGen(process.argv.includes('--check'));
}
