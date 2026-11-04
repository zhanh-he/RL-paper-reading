import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve } from 'node:path';
const root = resolve(import.meta.dirname, '../..');
const source = resolve(root, 'music-gen/reward-regime-study/naturalness/runs/2026-11-04-psychoacoustic');
const bytes = await readFile(resolve(source, 'audit.json'), 'utf8');
const data = JSON.parse(bytes), features = JSON.parse(await readFile(resolve(source, 'features.json')));
const old = JSON.parse(await readFile(resolve(root, 'music-gen/reward-regime-study/naturalness/runs/2026-11-04/audit.json')));
const bindings = JSON.parse(await readFile(resolve(root, 'platform/site/demos/audio-encoding.json'))).files;
assert.equal(data.samples.length, 47); assert.equal(features.samples.length, 47);
assert.equal(data.metadata.human_labels_count, 0); assert.equal(data.checks.human_labels_count, 0);
assert.equal(new Set(data.samples.map(s => s.id)).size, 47);
assert.equal(data.metadata.annotation_compatible_versions[0], old.metadata.version);
assert.ok(!/\/Users\/|\/home\/|\/storage\/|\blab\d{4}\b/.test(bytes));
const finite = (v) => { if (typeof v === 'number') assert.ok(Number.isFinite(v)); else if (v && typeof v === 'object') Object.values(v).forEach(finite); };
finite(data); finite(features);
for (const row of data.samples) {
  const r = features.samples.find(s => s.id === row.id); assert.ok(r);
  assert.equal(row.excerpt_sha256, r.audio_sha256); assert.equal(row.human_rating, null);
  assert.equal(row.signal.out_of_range_fraction, 0);
  assert.equal(row.psycho.valid, r.auditory.valid && r.roughness.valid);
  if (row.psycho.valid) {
    assert.equal(row.psycho.roughness70, r.roughness.levels['70'].asper.mean);
    assert.equal(row.psycho.modulation_depth, r.auditory.modulation_depth_20_300);
    assert.equal(row.psycho.resonance_protected, r.auditory.harmonic_veto_resonance_proxy);
    assert.ok(r.roughness.levels['60'] && r.roughness.levels['80']);
  } else assert.equal(row.psycho.roughness70, null);
  const file = 'platform/site/demos/' + row.audio;
  assert.equal(createHash('sha256').update(await readFile(resolve(root, file))).digest('hex'), row.mp3_sha256);
  assert.equal(bindings[file].source_audio_sha256, row.excerpt_sha256);
}
for (const row of old.samples) {
  const same = data.samples.find(s => s.id === row.id);
  assert.equal(same.excerpt_sha256, row.excerpt_sha256); assert.equal(same.mp3_sha256, row.mp3_sha256);
}
for (const group of ['psy_muse_baseline', 'psy_ref_vibeace']) {
  const rows = data.samples.filter(s => s.pair_group === group);
  assert.equal(rows.length, 8);
  assert.ok(Math.max(...rows.map(s => s.signal.rms_dbfs)) - Math.min(...rows.map(s => s.signal.rms_dbfs)) < 1e-5);
}
for (const p of data.listening_pairs) assert.ok(data.samples.some(s => s.id === p.a) && data.samples.some(s => s.id === p.b));
const dest = resolve(root, 'platform/site/demos/data/psychoacoustic-audit.json');
if (process.argv.includes('--check')) assert.equal(await readFile(dest, 'utf8'), bytes); else await writeFile(dest, bytes);
console.log('Psychoacoustic audit checked: 47 original hashes / MP3s, 3 SPL levels + raw assumption, 25 legacy ratings-compatible files.');
