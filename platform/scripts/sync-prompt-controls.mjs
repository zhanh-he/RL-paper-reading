import assert from 'node:assert/strict';
import {readFile,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {resolve} from 'node:path';
const root=resolve(import.meta.dirname,'../..'),run=resolve(root,'music-gen/reward-regime-study/naturalness/runs/2026-11-05-controls');
const bytes=await readFile(resolve(run,'audit.json'),'utf8'),data=JSON.parse(bytes),raw=JSON.parse(await readFile(resolve(run,'raw-scores.json'))),quality=JSON.parse(await readFile(resolve(run,'audiobox-scores.json'))),pcm=JSON.parse(await readFile(resolve(run,'muse-pcm-checks.json')));
const source=await readFile(resolve(root,'music-gen/reward-regime-study/naturalness/runs/2026-11-04-suno-real/audit.json'));
const bindings=JSON.parse(await readFile(resolve(root,'platform/site/demos/audio-encoding.json'))).files;
const sha=b=>createHash('sha256').update(b).digest('hex');const valid=v=>{if(typeof v==='number')assert.ok(Number.isFinite(v));else if(v&&typeof v==='object')Object.values(v).forEach(valid);};
assert.equal(data.samples.length,60);assert.equal(data.tracks.length,32);assert.equal(raw.samples.length,60);assert.equal(quality.samples.length,60);assert.equal(data.listening_pairs.length,14);
assert.equal(data.metadata.version,'prompt-control-20261105-v1');assert.equal(data.metadata.human_labels_count,0);assert.equal(data.checks.rl_training_runs,0);assert.equal(data.checks.audiobox_same_wav_scores,60);assert.equal(data.checks.reference_suno_audit_sha256,sha(source));
assert.ok(!/\/Users\/|\/home\/|\/storage\/|\blab\d{4}\b/.test(bytes));valid(data);valid(raw);
assert.equal(new Set(data.samples.map(s=>s.id)).size,60);assert.equal(new Set(data.tracks.map(s=>s.id)).size,32);
const originals=new Map(raw.samples.map(s=>[s.id,s])),aq=new Map(quality.samples.map(s=>[s.id,s])),cal=data.analysis.feature_distance_calibration;
const publicFile=async (url,expected,original)=>{assert.ok(url.endsWith('.mp3'));const file='platform/site/demos/'+url.replace(/^\.\//,'');assert.equal(sha(await readFile(resolve(root,file))),expected);assert.equal(bindings[file].sha256,expected);assert.equal(bindings[file].source_audio_sha256,original);};
for(const s of data.samples){
 const r=originals.get(s.id),q=aq.get(s.id);assert.ok(r&&q);assert.equal(s.human_rating,null);assert.equal(s.transform,'none');assert.equal(s.kind,'actual_generated_control');assert.equal(s.excerpt_sha256,r.audio_sha256);assert.equal(s.excerpt_sha256,q.audio_sha256);
 assert.equal(s.psycho.roughness70,r.roughness70.asper.mean);assert.equal(s.psycho.roughness70_p95,r.roughness70.asper.p95);assert.equal(s.psycho.modulation_depth,r.auditory.modulation_depth_20_300);assert.equal(s.psycho.resonance_protected,r.auditory.harmonic_veto_resonance_proxy);assert.equal(s.audiobox.PQ,q.scores.PQ);
 assert.equal(s.psycho.roughness60,null);assert.equal(s.psycho.roughness80,null);assert.equal(s.duration_s,10);assert.equal(s.sample_rate,48000);
 const vector=[s.psycho.modulation_depth,Math.log(s.psycho.roughness70+1e-4),Math.log(s.psycho.resonance_protected+1e-4)];const distance=Math.sqrt(vector.reduce((sum,v,i)=>sum+((v-cal.center[i])/cal.iqr[i])**2,0)/3);assert.ok(Math.abs(distance-s.psycho.suno_descriptor_distance)<1e-10);
 for(const t of Object.values(s.timeline)){assert.equal(t.time_s.length,t.value.length);assert.ok(t.time_s.every(v=>v>=0&&v<=10));}
 await publicFile(s.audio,s.mp3_sha256,s.excerpt_sha256);
}
for(const t of data.tracks){
 const windows=data.samples.filter(s=>s.source_case_id===t.id);assert.deepEqual(windows.map(s=>s.excerpt_start_s),t.duration_s>=50?[10,40]:[(t.duration_s-10)/2]);
 for(const key of ['modulation_depth','roughness70','resonance_protected'])assert.ok(Math.abs(windows.reduce((sum,s)=>sum+s.psycho[key],0)/windows.length-t.window_mean[key])<1e-12);
 assert.ok(Math.abs(windows.reduce((sum,s)=>sum+s.audiobox.PQ,0)/windows.length-t.audiobox_window_mean_PQ)<1e-12);await publicFile(t.full_audio,t.full_mp3_sha256,t.audio_sha256);
}
for(const n of data.checks.same_seed_noise_checks){assert.equal(n.all_initial_noise_blocks_identical,true);assert.equal(n.blocks_per_case,30);}
assert.equal(pcm.native_pcm_identical,true);assert.equal(pcm.native_pcm_max_abs_difference,0);assert.equal(pcm.override1p5_matches_native_pcm,true);assert.equal(data.checks.muse_native_and_renderer_checks.native_different_arguments_identical_audio,true);
const muse=data.tracks.filter(t=>t.id.startsWith('muse_')&&t.arm==='base');assert.equal(muse.length,3);assert.equal(new Set(muse.map(t=>t.audio_tokens_sha256)).size,1);assert.deepEqual(muse.map(t=>t.actual_guidance_scale),[1,1.5,2]);assert.ok(muse.every(t=>t.same_decoder_random_draws_as_first_identical_codes===true));
for(const p of data.listening_pairs)assert.ok(data.samples.some(s=>s.id===p.a)&&data.samples.some(s=>s.id===p.b));
const dest=resolve(root,'platform/site/demos/data/prompt-control-audit.json');if(process.argv.includes('--check'))assert.equal(await readFile(dest,'utf8'),bytes);else await writeFile(dest,bytes);
console.log('Prompt controls checked:32 real outputs /60 fixed windows /92 MP3 bindings /60 exact-WAV quality scores; nativeAPI PCM no-op and actual renderer controls verified;0 human labels.');
