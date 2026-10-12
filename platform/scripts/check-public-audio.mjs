import assert from 'node:assert/strict';
import { readdir, readFile, stat } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve, extname } from 'node:path';

const root = resolve(import.meta.dirname, '../..');
const site = resolve(root, 'platform/site');
const forbidden = new Set(['.wav', '.flac', '.ogg', '.m4a', '.aac', '.aiff', '.aif']);
async function* files(dir) {
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    if (['.git', 'node_modules', '.venv', '__pycache__'].includes(entry.name)) continue;
    const path = resolve(dir, entry.name);
    if (entry.isDirectory()) yield* files(path); else yield path;
  }
}
for await (const path of files(root)) assert.ok(!forbidden.has(extname(path).toLowerCase()), `MP3-only policy: ${path}`);
const manifest = JSON.parse(await readFile(resolve(site, 'demos/audio-encoding.json')));
for (const [relative, binding] of Object.entries(manifest.files)) {
  assert.ok(relative.endsWith('.mp3'));
  const path = resolve(root, relative);
  const hash = createHash('sha256').update(await readFile(path)).digest('hex');
  assert.equal(hash, binding.sha256); assert.equal((await stat(path)).size, binding.bytes);
  assert.match(binding.source_audio_sha256, /^[a-f0-9]{64}$/);
}
async function checkJson(value, key) {
  if (Array.isArray(value)) { for (const item of value) await checkJson(item); }
  else if (value && typeof value === 'object') { for (const [k, v] of Object.entries(value)) await checkJson(v, k); }
  else if (typeof value === 'string' && key !== 'source_path' && /^(?:\.\/)?audio\//.test(value)) {
    assert.ok(value.endsWith('.mp3'), `Public playback must be MP3: ${value}`);
    await stat(resolve(site, 'demos', value));
  }
}
for await (const path of files(resolve(site, 'demos'))) {
  if (extname(path) === '.json') await checkJson(JSON.parse(await readFile(path)));
}
console.log(`MP3-only policy checked; ${Object.keys(manifest.files).length} legacy listening copies hash-verified.`);
