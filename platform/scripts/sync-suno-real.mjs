import assert from 'node:assert/strict';
import {readFile, writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {resolve} from 'node:path';
const root=resolve(import.meta.dirname,'../..'), source=resolve(root,'music-gen/reward-regime-study/naturalness/runs/2026-11-04-suno-real');
const bytes=await readFile(resolve(source,'audit.json'),'utf8'), data=JSON.parse(bytes), features=JSON.parse(await readFile(resolve(source,'features.json')));
const bindings=JSON.parse(await readFile(resolve(root,'platform/site/demos/audio-encoding.json'))).files;
assert.equal(data.samples.length,48);assert.equal(features.samples.length,48);assert.equal(features.status,'complete');
assert.equal(data.metadata.human_labels_count,0);assert.equal(data.checks.model_training_runs,0);assert.equal(data.checks.same_input_verified_pairs,0);
assert.equal(new Set(data.samples.map(s=>s.id)).size,48);assert.equal(new Set(data.samples.map(s=>s.source_original_sha256)).size,16);assert.equal(new Set(data.samples.map(s=>s.source_family)).size,8);
assert.ok(!/\/Users\/|\/home\/|\/storage\/|\blab\d{4}\b/.test(bytes));
const checkFinite=v=>{if(typeof v==='number')assert.ok(Number.isFinite(v));else if(v&&typeof v==='object')Object.values(v).forEach(checkFinite);};checkFinite(data);checkFinite(features);
for(const s of data.samples){
 const f=features.samples.find(r=>r.id===s.id);assert.equal(s.kind,'real_suno_output');assert.equal(s.transform,'none');assert.equal(s.same_input_verified,false);assert.equal(s.human_rating,null);
 assert.equal(s.excerpt_sha256,f.audio_sha256);assert.equal(s.psycho.roughness70,f.roughness.levels['70'].asper.mean);assert.equal(s.psycho.modulation_depth,f.auditory.modulation_depth_20_300);assert.equal(s.psycho.resonance_protected,f.auditory.harmonic_veto_resonance_proxy);
 assert.ok(Math.abs(s.excerpt_start_s-(s.source_duration_s-10)*s.source_fraction)<.00051);assert.equal(s.duration_s,10);
 assert.equal(s.source_record.revision,data.checks.source_dataset_revision);assert.ok(s.source_record.tar_member.endsWith('.mp3'));assert.ok(s.publisher_style.length);
 assert.ok(!('audiobox' in s)&&!('songeval' in s));
 for(const series of Object.values(s.timeline)){assert.equal(series.time_s.length,series.value.length);assert.ok(series.time_s.every(t=>t>=0&&t<=10));}
 const file='platform/site/demos/'+s.audio;assert.equal(createHash('sha256').update(await readFile(resolve(root,file))).digest('hex'),s.mp3_sha256);assert.equal(bindings[file].source_audio_sha256,s.excerpt_sha256);
}
for(const song of new Set(data.samples.map(s=>s.song_id))){const rows=data.samples.filter(s=>s.song_id===song);assert.deepEqual(rows.map(s=>s.source_fraction),[.2,.5,.8]);}
for(const pair of data.listening_pairs){const a=data.samples.find(s=>s.id===pair.a),b=data.samples.find(s=>s.id===pair.b);assert.ok(a&&b);assert.equal(a.source_family,b.source_family);assert.equal(a.source_fraction,b.source_fraction);assert.notEqual(a.track_index,b.track_index);}
const dest=resolve(root,'platform/site/demos/data/suno-real-audit.json');if(process.argv.includes('--check'))assert.equal(await readFile(dest,'utf8'),bytes);else await writeFile(dest,bytes);
console.log('Suno real audit checked:16 original songs /8 families /48 unmodified excerpts /24 family comparisons;0 labels; exact score-to-MP3 hash bindings.');
