import assert from 'node:assert/strict';
import {readFile,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {resolve} from 'node:path';
const root=resolve(import.meta.dirname,'../..');
const source=resolve(root,'music-gen/reward-regime-study/naturalness/runs/2026-11-05-listening/review.json');
const bytes=await readFile(source,'utf8'),data=JSON.parse(bytes),frozenBytes=await readFile(resolve(root,'music-gen/reward-regime-study/naturalness/runs/2026-11-04-suno-real/audit.json'));
const old=JSON.parse(frozenBytes);assert.equal(createHash('sha256').update(frozenBytes).digest('hex'),data.source_audit_sha256);
assert.equal(data.shortlist.length,5);assert.equal(data.traces.length,48);assert.equal(data.checks.maximum_scalar_difference,0);assert.equal(data.metadata.human_labels_count,0);
assert.ok(!/\/Users\/|\/home\/|\/storage\/|\blab\d{4}\b/.test(bytes));
const by=new Map(old.samples.map(s=>[s.id,s]));
assert.equal(new Set(data.traces.map(s=>s.id)).size,48);assert.equal(new Set(data.shortlist.map(s=>s.id)).size,5);
for(const t of data.traces){const s=by.get(t.id);assert.ok(s);assert.equal(t.audio_sha256,s.excerpt_sha256);assert.equal(t.time_s.length,t.value.length);assert.ok(t.time_s.length>900);assert.ok(t.time_s.every((v,i)=>Number.isFinite(v)&&v>=0&&v<=10&&(i===0||v>t.time_s[i-1])));assert.ok(t.value.every(v=>Number.isFinite(v)&&v>=0&&v<=1));assert.ok(Math.max(...t.value)>0);}
for(const p of data.shortlist){assert.ok(by.has(p.a)&&by.has(p.b));assert.ok(p.purpose&&p.caveat&&p.observation);for(const c of p.checkpoints){assert.ok([p.a,p.b].includes(c.sample_id));assert.ok(c.clip_s>=0&&c.clip_s<=10);assert.ok(Math.abs(by.get(c.sample_id).excerpt_start_s+c.clip_s-c.source_s)<1e-8);}}
const dest=resolve(root,'platform/site/demos/data/listening-review.json');if(process.argv.includes('--check'))assert.equal(await readFile(dest,'utf8'),bytes);else await writeFile(dest,bytes);
console.log('Listening review checked:5 exploratoryrealSuno pairs,48 full10msresonance traces,originalscalar/audio hashes unchanged.');
