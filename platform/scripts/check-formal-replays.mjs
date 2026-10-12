import assert from 'node:assert/strict';
import { readFile, access } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { createHash } from 'node:crypto';

const base = fileURLToPath(new URL('../site/demos/', import.meta.url));
const data = JSON.parse(await readFile(path.join(base, 'formal-9to1-runs.json')));
if (data.schema_version !== 3) throw new Error('Expected recovered formal registry v3');
assert.equal(data.train_conditions, 270); assert.equal(data.valid_conditions, 30); assert.equal(data.sealed_test_conditions, 121);
assert.equal(data.arms.length, 12);
assert.deepEqual(data.visual_protocol.waveform_bounds, [-1, 1]);
assert.deepEqual(data.visual_protocol.frequency_hz, [0, 24000]);
assert.deepEqual(data.visual_protocol.decibels, [-80, 0]);
assert.deepEqual(data.examples.map((r) => r.index), [0, 10, 20]);
const baselineHashes = new Map();
let clips = 0;
for (const arm of data.arms) {
  assert.ok(arm.id && arm.model && arm.reward && arm.lr);
  assert.ok(['2e-5', '1e-4', '3e-4'].includes(arm.lr));
  if (arm.model === 'Muse') {
    assert.equal(arm.standalone_inference, false);
    assert.equal(arm.validation_generation, false);
    assert.match(arm.job_id, /^\d+\.gadi-pbs$/);
    assert.equal(arm.priority, 1023);
    assert.deepEqual(arm.long_replays, {});
  }
  assert.equal(arm.train_curve.length, arm.verified_train_steps);
  arm.train_curve.forEach((r, i) => { assert.equal(r.step, i + 1); assert.ok(Number.isFinite(r.reward)); assert.ok(Number.isFinite(r.sampled_kl)); });
  if (arm.status === 'verified-complete') {
    assert.equal(arm.verified_train_steps, 270);
    assert.equal(arm.validation['0'].n, 30); assert.equal(arm.validation['270'].n, 30);
  }
  if (arm.status.includes('failed')) assert.ok(arm.failure_reason);
  for (const [step, rows] of Object.entries(arm.long_replays)) {
    assert.ok(Number(step) === 0 || arm.checkpoints.includes(Number(step)), `Unknown checkpoint ${arm.id}/${step}`);
    assert.equal(new Set(rows.map((r) => r.index)).size, rows.length);
    for (const row of rows) {
      assert.ok([0, 10, 20].includes(row.index)); assert.equal(row.seed, 5101 + row.index);
      assert.match(row.sha256, /^[a-f0-9]{64}$/);
      assert.ok(row.audio.endsWith(`${row.sha256.slice(0, 12)}.mp3`));
      assert.match(row.source_audio_sha256, /^[a-f0-9]{64}$/);
      assert.ok(row.audio.startsWith('https://github.com/zhanh-he/RL-paper-reading/releases/download/formal-cmi-pref-20261012/'));
      if (arm.model === 'Muse') { assert.equal(row.inference_backend, 'vllm'); assert.equal(row.inference_dtype, 'bfloat16'); }
      assert.equal(row.reaches_30_seconds, row.seconds >= 30);
      assert.ok(Number.isFinite(row.signal.peak));
      for (const type of ['wave', 'spectrum']) {
        const asset = row[type]; assert.ok(!asset.includes('..')); await access(path.join(base, asset));
        assert.equal(createHash('sha256').update(await readFile(path.join(base, asset))).digest('hex'), row[`${type}_sha256`]);
      }
      if (Number(step) === 0) {
        const key = `${arm.model}/${row.index}`;
        if (baselineHashes.has(key)) assert.equal(row.sha256, baselineHashes.get(key), `Baseline mismatch ${key}`);
        else baselineHashes.set(key, row.sha256);
      }
      if (row.reward) { assert.ok(row.scored_audio_sha256); assert.ok(Number.isFinite(row.reward.mean)); }
      clips++;
    }
  }
}
assert.equal(data.superseded_arms.length, 6);
console.log(`Formal registry checked: ${data.arms.length} arms, ${clips} clip bindings, fixed per-model baseline.`);
