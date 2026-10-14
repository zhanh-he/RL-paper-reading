import assert from 'node:assert/strict';
import fs from 'node:fs';
const source = new URL('../../music-gen/reward-regime-study/protocol.json', import.meta.url);
const target = new URL('../site/demos/data/reward-regime-protocol.json', import.meta.url);
const value = JSON.parse(fs.readFileSync(source));
assert.equal(value.training_authorized, true);
assert.equal(value.runs.length, 6);
assert.deepEqual(value.design.lr_grid, ['3e-4']);
assert.deepEqual(value.design.seeds, [42]);
assert.equal(new Set(value.runs.map((r) => `${r.model}-${r.reward}`)).size, 6);
assert.ok(value.runs.every((r) => ['yue2', 'muse'].includes(r.model)));
assert.equal(value.rewards.length, 3);
assert.deepEqual(value.rewards.map((r) => r.id), ['richness', 'songbench', 'songbench-arrangement']);
assert.equal(value.reward_selection.third_reward, 'SongBench-Arrangement');
assert.ok(value.models.every((m) => m.mos === null));
const rendered = JSON.stringify(value, null, 2) + '\n';
if (process.argv.includes('--check')) assert.equal(fs.readFileSync(target, 'utf8'), rendered);
else fs.writeFileSync(target, rendered);
const progress = JSON.parse(fs.readFileSync(new URL('../../music-gen/reward-regime-study/progress.json', import.meta.url)));
for (const run of progress.runs) {
  assert.equal(run.steps, run.trace.length);
  assert.ok(run.trace.every((p, i) => p.step === i+1 && Number.isFinite(p.reward) && Number.isFinite(p.kl)));
  assert.ok(run.validation.every((p) => p.n === 30 && Number.isFinite(p.mean) && p.receipt_sha256.length === 64));
}
const progressTarget = new URL('../site/demos/data/reward-regime-progress.json', import.meta.url);
const progressJSON = JSON.stringify(progress, null, 2)+'\n';
if (process.argv.includes('--check')) assert.equal(fs.readFileSync(progressTarget, 'utf8'), progressJSON);
else fs.writeFileSync(progressTarget, progressJSON);
console.log('SP/AI protocol: six approved arms, LR3e-4, verified trace/all-30 validation, MOS not measured.');
