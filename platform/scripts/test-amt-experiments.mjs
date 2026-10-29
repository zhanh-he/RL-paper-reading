import assert from 'node:assert/strict';
import test from 'node:test';
import { validateExperiments } from './sync-amt-experiments.mjs';
import { amtFixture } from './test-fixtures/amt-experiments.mjs';

test('empty AMT workspace is valid; real notes can be silent', () => {
  validateExperiments({ schema_version: 1, updated_at: '2026-10-29', experiments: [], datasets: [], models: [], rewards: [] });
  const data = amtFixture(); data.experiments[0].stages[1].outputs['sample-a'].notes = [];
  validateExperiments(data);
});
test('planned results and measured stages without receipts are rejected', () => {
  let data = amtFixture(); data.experiments[0].stages[1].status = 'planned';
  assert.throws(() => validateExperiments(data), /planned stages/);
  data = amtFixture(); delete data.experiments[0].receipt;
  assert.throws(() => validateExperiments(data), /receipt/);
});
test('duplicate/missing baselines, cross-sample outputs and incorrect F1 units fail', () => {
  let data = amtFixture(); data.experiments[0].stages[1].step = 0;
  assert.throws(() => validateExperiments(data), /duplicate optimizer/);
  data = amtFixture(); data.experiments[0].stages.shift();
  assert.throws(() => validateExperiments(data), /baseline step 0/);
  data = amtFixture(); data.experiments[0].stages[1].outputs['unknown-song'] = { notes: [] };
  assert.throws(() => validateExperiments(data), /unknown sample/);
  data = amtFixture(); data.experiments[0].stages[1].evaluation.frame = 60;
  assert.throws(() => validateExperiments(data), /fraction/);
});
test('public files, note timing, unsafe paths and audio formats are checked', () => {
  assert.throws(() => validateExperiments(amtFixture(), () => false), /file not found/);
  let data = amtFixture(); data.experiments[0].samples[0].input.audio = '../private.mp3';
  assert.throws(() => validateExperiments(data), /relative path/);
  data = amtFixture(); data.experiments[0].samples[0].input.audio = 'audio/private.wav';
  assert.throws(() => validateExperiments(data), /unsupported public format/);
  data = amtFixture(); data.experiments[0].samples[0].reference.notes[0].end = 10;
  assert.throws(() => validateExperiments(data), /note timing/);
});
