import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve } from 'node:path';

const root = resolve(import.meta.dirname, '../..');
const run = 'music-gen/reward-regime-study/naturalness/runs/2026-11-04/';
const check = process.argv.includes('--check');
const read = async (name) => JSON.parse(await readFile(resolve(root, run, name), 'utf8'));
const raw = await read('audit.json');
const learned = await read('learned-scores.json');
const song = await read('songeval-scores.json');
const signal = await read('signal-scores.json');
const binding = JSON.parse(await readFile(resolve(root, 'platform/site/demos/audio-encoding.json'))).files;
assert.equal(raw.samples.length, 25);
assert.equal(raw.metadata.human_labels_count, 0);
assert.equal(raw.label_agreement.pair_accuracy, null);
assert.equal(new Set(raw.samples.map(s => s.id)).size, 25);
const assertFinite = (value) => {
  if (typeof value === 'number') assert.ok(Number.isFinite(value));
  else if (Array.isArray(value)) value.forEach(assertFinite);
  else if (value && typeof value === 'object') Object.values(value).forEach(assertFinite);
};
assertFinite(raw);
for (const row of raw.samples) {
  assert.equal(row.human_rating, null);
  for (const result of [learned.samples.find(s => s.id === row.id), song.samples.find(s => s.id === row.id), signal.rows.find(s => s.id === row.id)]) {
    assert.equal(result.audio_sha256, row.excerpt_sha256, row.id);
  }
  assert.deepEqual(row.audiobox, learned.samples.find(s => s.id === row.id).audiobox);
  assert.equal(row.pam.score, learned.samples.find(s => s.id === row.id).pam.PAM);
  for (const key of ['Coherence', 'Musicality', 'Memorability', 'Clarity', 'Naturalness']) {
    assert.equal(row.songeval[key], song.samples.find(s => s.id === row.id).scores[key]);
  }
  const relative = 'platform/site/demos/' + row.audio;
  assert.ok(relative.endsWith('.mp3'));
  const hash = createHash('sha256').update(await readFile(resolve(root, relative))).digest('hex');
  assert.equal(hash, row.mp3_sha256);
  assert.equal(binding[relative].source_audio_sha256, row.excerpt_sha256);
  assert.match(row.source_original_sha256, /^[a-f0-9]{64}$/);
}
const bytes = await readFile(resolve(root, run, 'audit.json'), 'utf8');
assert.ok(!/\/Users\/|\/home\/|\/storage\/|\blab\d{4}\b/.test(bytes), 'Private environment data in public snapshot');
const target = resolve(root, 'platform/site/demos/data/naturalness-audit.json');
if (check) assert.equal(await readFile(target, 'utf8'), bytes);
else { await mkdir(resolve(root, 'platform/site/demos/data'), { recursive: true }); await writeFile(target, bytes); }
console.log(`Naturalness audit ${check ? 'checked' : 'built'}: 25 exact scorer hashes, 25 MP3 bindings, 0 invented human labels.`);
