import assert from 'node:assert/strict';
import test from 'node:test';
import { validateExperiments } from './sync-amt-experiments.mjs';
import { amtFixture } from './test-fixtures/amt-experiments.mjs';
import fs from 'node:fs';

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

test('measured evaluation controls use conditions and explicit metric semantics, not steps', () => {
  const data = JSON.parse(fs.readFileSync(new URL('../data/amt/experiments.json', import.meta.url)));
  validateExperiments(data);
  assert.equal(data.experiments.find(run => run.id === 'metric-blindspots').conditions.length, 5);
  const index = data.experiments.findIndex(run => run.id === 'metric-blindspots');
  let changed = structuredClone(data);
  changed.experiments[index].conditions[0].status = 'planned';
  assert.throws(() => validateExperiments(changed), /unmeasured conditions/);
  changed = structuredClone(data); changed.experiments[index].trace = [{step:0,reward:1}];
  assert.throws(() => validateExperiments(changed), /invent optimizer/);
  changed = structuredClone(data); changed.experiments[index].conditions[0].evaluation.values.class_onset = 100;
  assert.throws(() => validateExperiments(changed), /fraction/);
  changed = structuredClone(data); changed.experiments[index].conditions[0].evaluation.values.unknown = 1;
  assert.throws(() => validateExperiments(changed), /unknown/);
  changed = structuredClone(data); changed.experiments[index].conditions[0].outputs['wrong-sample'] = {notes:[]};
  assert.throws(() => validateExperiments(changed), /unknown sample/);
  changed = structuredClone(data); changed.reported_results[0].rwc = 101;
  assert.throws(() => validateExperiments(changed), /reported F1/);
});

test('real score benchmarks retain complete cohorts, scoped GRPO budgets and source-backed score assets', () => {
  const data = JSON.parse(fs.readFileSync(new URL('../data/amt/experiments.json', import.meta.url)));
  const study = JSON.parse(fs.readFileSync(new URL('../data/amt/score-study.json', import.meta.url)));
  const duet = data.experiments.find(run => run.id === 'asap-duet-benchmark');
  assert.equal(duet.conditions.length, 3);
  assert.equal(duet.samples.length, 2);
  for (const condition of duet.conditions) {
    assert.equal(condition.evaluation.n, 8);
    for (const sample of duet.samples) assert.ok(condition.outputs[sample.id].score_image.endsWith('.svg'));
  }
  assert.equal(study.results.grpo.rollouts, 50);
  assert.equal(study.results.grpo.optimizer_steps, 40);
  assert.equal(study.results.grpo.trace.filter(row => row.optimizer_update).length, 40);
  assert.equal(study.results.sheet.samples.length, 10);
  assert.ok(study.results.sheet.samples.every(row => row.status === 'complete'));
  assert.ok(study.results.duet.sft50.MeanER > study.results.duet.baseline.MeanER);
  assert.ok(study.results.duet.grpo50.MeanER > study.results.duet.baseline.MeanER);
  let changed = structuredClone(data);
  changed.experiments.find(run => run.id === duet.id).samples[0].reference.score_image = '../private.svg';
  assert.throws(() => validateExperiments(changed), /relative path/);
  changed = structuredClone(data);
  changed.experiments.find(run => run.id === duet.id).samples[0].reference.musicxml = 'amt/demos/score-examples/private.html';
  assert.throws(() => validateExperiments(changed), /unsupported public format/);
});
