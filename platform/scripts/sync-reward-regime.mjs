import assert from 'node:assert/strict';
import fs from 'node:fs';
const source = new URL('../../music-gen/reward-regime-study/protocol.json', import.meta.url);
const target = new URL('../site/demos/data/reward-regime-protocol.json', import.meta.url);
const value = JSON.parse(fs.readFileSync(source));
assert.equal(value.training_authorized, true);
assert.equal(value.runs.length, 15);
assert.deepEqual(value.design.lr_grid, ['3e-4']);
assert.deepEqual(value.design.seeds, [42]);
assert.equal(new Set(value.runs.map((r) => `${r.model}-${r.reward}`)).size, 15);
assert.ok(value.runs.every((r) => ['yue2', 'muse', 'heartmula', 'diffrhythm2', 'songbloom'].includes(r.model)));
assert.equal(value.rewards.length, 3);
assert.deepEqual(value.rewards.map((r) => r.id), ['richness', 'songbench', 'songbench-arrangement']);
assert.equal(value.reward_selection.third_reward, 'SongBench-Arrangement');
assert.ok(value.models.every((m) => m.mos === null));
assert.deepEqual(value.models.map((m) => m.id), ['yue2', 'heartmula', 'diffrhythm2', 'songbloom', 'muse', 'private', 'acestep']);
assert.ok(value.models.every((m) => m.team_listening_estimate === null ||
  (Number.isFinite(m.team_listening_estimate) && m.team_listening_estimate >= 1 && m.team_listening_estimate <= 5)));
assert.match(value.expansion.listening_estimates, /not formal/);
assert.equal(value.models.find((m) => m.id === 'private').source, null);
assert.equal(value.expansion.new_training_submitted, true);
const rendered = JSON.stringify(value, null, 2) + '\n';
if (process.argv.includes('--check')) assert.equal(fs.readFileSync(target, 'utf8'), rendered);
else fs.writeFileSync(target, rendered);
const progress = JSON.parse(fs.readFileSync(new URL('../../music-gen/reward-regime-study/progress.json', import.meta.url)));
const startup = JSON.parse(fs.readFileSync(new URL('../../music-gen/reward-regime-study/runs/2026-10-14-open-head-startup/progress.json', import.meta.url)));
assert.equal(startup.complete, false); assert.equal(startup.steps, 5);
progress.runs.push(startup);
progress.updated_at = startup.snapshot_at;
for (const run of progress.runs) {
  assert.equal(run.steps, run.trace.length);
  assert.ok(run.trace.every((p, i) => p.step === i+1 && Number.isFinite(p.reward) && Number.isFinite(p.kl)));
  assert.ok(run.validation.every((p) => p.n === 30 && Number.isFinite(p.mean) && p.receipt_sha256.length === 64));
}
const progressTarget = new URL('../site/demos/data/reward-regime-progress.json', import.meta.url);
const progressJSON = JSON.stringify(progress, null, 2)+'\n';
if (process.argv.includes('--check')) assert.equal(fs.readFileSync(progressTarget, 'utf8'), progressJSON);
else fs.writeFileSync(progressTarget, progressJSON);
console.log('SP/AI: 15 registered arms; completed, updating, sequential and preparation states distinguished; five fixed listening cases.');
const examples = JSON.parse(fs.readFileSync(new URL('../site/demos/data/reward-regime-examples.json', import.meta.url)));
assert.deepEqual(examples.display_indices, [0,10,20,30,31]);
assert.equal(examples.inputs.length, 32);
const [original, two, five] = [0,30,31].map(i => examples.inputs.find(r => r.index === i));
assert.equal(two.lyrics, original.lyrics); assert.equal(five.lyrics, original.lyrics);
assert.equal(two.seed, original.seed); assert.equal(five.seed, original.seed);
assert.equal(two.requested_count, 2); assert.equal(five.requested_count, 5);
assert.equal(two.original_style, original.style); assert.equal(five.original_style, original.style);
assert.match(two.scope, /excluded/); assert.match(five.scope, /excluded/);
const recoverySource=new URL('../../music-gen/reward-regime-study/recovery.json',import.meta.url);
if(fs.existsSync(recoverySource)) {
  const recovery=JSON.parse(fs.readFileSync(recoverySource));
  assert.equal(recovery.schema,1);
  assert.ok(recovery.audits.every((r)=>r.n===30));
  for(const receipt of recovery.validation_receipts) {
    assert.equal(receipt.n,30);assert.equal(receipt.rows.length,30);
    assert.ok(receipt.rows.every((r,i)=>r.index===i&&r.seed===5101+i&&r.source_audio_sha256.length===64));
    const measured=progress.runs.find((r)=>r.reward===receipt.reward).validation.find((r)=>r.step===receipt.step);
    const mean=receipt.rows.reduce((n,r)=>n+r.metrics.objective,0)/30;
    assert.ok(Math.abs(mean-measured.mean)<1e-10,'Full validation receipt/curve mismatch');
  }
  for(const c of recovery.badcases) {
    for(const [side,step] of [['baseline',0],['candidate',c.step]]) {
      const receipt=recovery.validation_receipts.find((r)=>r.reward===c.reward&&r.step===step);
      assert.equal(c[side].source_audio_sha256,receipt.rows[c.index].source_audio_sha256);
    }
    assert.ok(c.candidate.metrics.objective>c.baseline.metrics.objective);
  }
  const recoveryTarget=new URL('../site/demos/data/reward-regime-recovery.json',import.meta.url);
  const encoded=JSON.stringify(recovery,null,2)+'\n';
  if(process.argv.includes('--check'))assert.equal(fs.readFileSync(recoveryTarget,'utf8'),encoded);
  else fs.writeFileSync(recoveryTarget,encoded);
}
