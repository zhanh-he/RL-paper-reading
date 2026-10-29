import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { validateMusicGen, sha256 } from './sync-musicgen-experiments.mjs';

const source = new URL('../data/musicgen/experiments.json', import.meta.url);
const dataset = JSON.parse(await readFile(source, 'utf8'));
const copy = () => structuredClone(dataset);
test('all four planned factorial cells remain separate from engineering evidence', () => {
  validateMusicGen(copy());
  assert.equal(dataset.ablations.length, 4);
  assert(dataset.ablations.every(row => row.status === 'planned'));
  assert(dataset.evidence.every(row => row.kind === 'engineering_check'));
  assert.equal(dataset.training_results.length, 0);
});
test('public identities must be strings, not coerced nulls or numbers', () => {
  for (const value of [null, 123, false]) {
    const data = copy(); data.evidence[0].id = value;
    assert.throws(() => validateMusicGen(data), /public ID/);
  }
});
test('repeated factorial cells and false completion are rejected', () => {
  const repeated = copy(); repeated.ablations[1].score_plan = false;
  assert.throws(() => validateMusicGen(repeated), /factorial cell/);
  const completed = copy(); completed.ablations[0].status = 'completed';
  assert.throws(() => validateMusicGen(completed));
});
test('raw runtime fields, private paths, credentials and private implementation links are rejected', () => {
  for (const [key, value] of [['logs', 'raw log'], ['hostname', 'training-node'], ['checkpoint', 'model.pt']]) {
    const data = copy(); data.evidence[0][key] = value;
    assert.throws(() => validateMusicGen(data), /public fields/);
  }
  for (const value of ['/cache/secret/data', '/home/person/run', 'file:///private/report', 'https://github.com/example-private-team/internal-model', 'hf_1234567890123456789012345']) {
    const data = copy(); data.evidence[0].finding = value;
    assert.throws(() => validateMusicGen(data), /rejected/);
  }
});
test('engineering counts cannot be relabeled as quality evidence', () => {
  const data = copy(); data.evidence[0].kind = 'quality_measurement';
  assert.throws(() => validateMusicGen(data));
});
function measured() {
  const data = copy();
  data.training_results.push({ id:'example-result', experiment_id:'example-experiment', date:'2026-10-29', task:'vocal2accomp', evidence_kind:'training_measurement', status:'completed', config_id:'config-public-1', split_id:'song-split-1', seed:42, metrics:[{id:'heldout-token-nll', value:2.5, n:10, unit:'nats-per-token', direction:'lower_is_better', scope:'training_diagnostic'}], summary:'Schema fixture, never published as a real result.' });
  return data;
}
test('future results need explicit scope, finite values and sample counts', () => {
  validateMusicGen(measured());
  const quality = measured(); quality.training_results[0].metrics[0].scope = 'quality';
  assert.throws(() => validateMusicGen(quality), /cannot claim music quality/);
  for (const value of [NaN, Infinity]) {
    const data = measured(); data.training_results[0].metrics[0].value = value;
    assert.throws(() => validateMusicGen(data), /finite/);
  }
  const missing = measured(); missing.training_results[0].metrics[0].n = null;
  assert.throws(() => validateMusicGen(missing), /count/);
  missing.training_results[0].metrics[0].value = null;
  validateMusicGen(missing);
  const failed = measured(); failed.training_results[0].status = 'failed';
  assert.throws(() => validateMusicGen(failed), /Failed runs/);
});
test('public snapshot hashes bind the exact generated JSON', async () => {
  const publicText = await readFile(new URL('../site/musicgen/data/experiments.json', import.meta.url), 'utf8');
  const snapshot = JSON.parse(await readFile(new URL('../site/musicgen/data/snapshot.json', import.meta.url)));
  assert.equal(snapshot.source_sha256, sha256(await readFile(source, 'utf8')));
  assert.equal(snapshot.dataset_sha256, sha256(publicText));
  assert.deepEqual(JSON.parse(publicText), dataset);
});
