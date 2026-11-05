import assert from 'node:assert/strict';
import {readFile,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {resolve} from 'node:path';
const root=resolve(import.meta.dirname,'../..'),base=resolve(root,'music-gen/reward-regime-study/naturalness'),run=resolve(base,'runs/2026-11-05-contrast');
const bytes=await readFile(resolve(run,'audit.json'),'utf8'),data=JSON.parse(bytes),raw=JSON.parse(await readFile(resolve(run,'raw-scores.json'))),receipt=JSON.parse(await readFile(resolve(run,'receipt.json'))),quality=JSON.parse(await readFile(resolve(run,'audiobox-scores.json')));
const sha=x=>createHash('sha256').update(x).digest('hex'),finite=x=>typeof x==='number'&&Number.isFinite(x),bindings=JSON.parse(await readFile(resolve(root,'platform/site/demos/audio-encoding.json'))).files;
const encoding=JSON.parse(await readFile(resolve(run,'encoding-checks.json'))),encoded=new Map(encoding.checks.map(x=>[x.id,x]));assert.equal(encoding.checked,24);assert.equal(encoded.size,24);assert.ok(encoding.checks.every(x=>x.reencoding_identical===true));
assert.ok(!/\/Users\/|\/home\/|\/storage\/|\blab\d{4}\b/.test(bytes));
assert.equal(data.metadata.version,'contrast-followup-20261105-v1');assert.equal(data.contrast_followup,true);assert.equal(data.attribute_probe,true);assert.equal(data.metadata.human_labels_count,0);assert.equal(data.checks.rl_training_runs,0);
assert.equal(data.samples.length,16);assert.equal(data.tracks.length,8);assert.equal(data.listening_pairs.length,6);assert.equal(raw.samples.length,16);assert.equal(quality.samples.length,16);assert.equal(new Set(data.samples.map(s=>s.id)).size,16);
assert.equal(receipt.profile.batch_sha256,sha(await readFile(resolve(run,'batch.json'))));assert.equal(receipt.profile.shared_source_unchanged,true);assert.equal(receipt.profile.sham_controls.length,2);
for(const c of receipt.profile.sham_controls)for(const k of ['same_latent','same_pcm','same_noise'])assert.equal(c[k],true);
const originals=new Map(raw.samples.map(s=>[s.id,s])),qs=new Map(quality.samples.map(s=>[s.id,s]));
const publicFile=async(url,hash,source)=>{assert.ok(url.endsWith('.mp3'));const rel='platform/site/demos/'+url.replace(/^\.\//,'');assert.equal(sha(await readFile(resolve(root,rel))),hash);assert.equal(bindings[rel].sha256,hash);assert.equal(bindings[rel].source_audio_sha256,source);};
for(const s of data.samples){
 const r=originals.get(s.id),q=qs.get(s.id);assert.ok(r&&q);assert.equal(s.excerpt_sha256,r.audio_sha256);assert.equal(s.excerpt_sha256,q.audio_sha256);assert.equal(s.human_rating,null);assert.equal(s.transform,'none');assert.equal(s.duration_s,10);assert.equal(s.sample_rate,48000);
 assert.equal(encoded.get(s.id).original_sha256,s.excerpt_sha256);assert.equal(encoded.get(s.id).mp3_sha256,s.mp3_sha256);
 for(const [key,value] of Object.entries({modulation_depth:r.auditory.modulation_depth_20_300,roughness70:r.roughness70.asper.mean,resonance_protected:r.auditory.harmonic_veto_resonance_proxy}))assert.equal(s.psycho[key],value);
 const m=r.auditory.maps;
 for(const [key,lo,hi] of [['modulation_low',0,1500],['modulation_high',3000,8000]]){
  let den=0,num=0;m.center_hz.forEach((f,i)=>{if(f>=lo&&f<hi)m.time_s.forEach((_,j)=>{const e=m.band_energy_relative[i][j],w=e>.001?e**2:0;den+=w;num+=w*m.modulation_depth[i][j];});});assert.ok(Math.abs(s.psycho[key]-num/den)<1e-12);
 }
 assert.equal(s.audiobox.PQ,q.scores.PQ);assert.ok(finite(s.signal.rms_dbfs));
 for(const t of Object.values(s.timeline)){assert.equal(t.time_s.length,t.value.length);assert.ok(t.time_s.every(v=>v>=0&&v<=10));assert.ok(t.value.every(finite));}
 await publicFile(s.audio,s.mp3_sha256,s.excerpt_sha256);
}
for(const t of data.tracks){
 const windows=data.samples.filter(s=>s.source_case_id===t.id);assert.deepEqual(windows.map(s=>s.excerpt_start_s),[10,40]);assert.equal(t.lyric_prefix.equal,true);assert.equal(t.lyric_prefix.good_sha256,t.lyric_prefix.bad_sha256);assert.equal(t.initial_noise_sha256.length,30);assert.equal(t.same_seed_noise_common_prefix_matches,true);assert.equal(t.duration_s,60);
 assert.equal(encoded.get(t.id).original_sha256,t.audio_sha256);assert.equal(encoded.get(t.id).mp3_sha256,t.full_mp3_sha256);
 assert.deepEqual(t.initial_noise_sha256,data.tracks[0].initial_noise_sha256);
 assert.equal(t.actual_sampler_kwargs.cfg_strength,t.beta);
 for(const k of ['modulation_depth','modulation_low','modulation_high','roughness70','resonance_protected'])assert.ok(Math.abs(windows.reduce((a,s)=>a+s.psycho[k],0)/2-t.window_mean[k])<1e-12);
 await publicFile(t.full_audio,t.full_mp3_sha256,t.audio_sha256);
 if(t.sham){const b=data.tracks.find(x=>x.family===t.family&&x.beta===0);assert.equal(t.audio_pcm_sha256,b.audio_pcm_sha256);assert.equal(t.latent_sha256,b.latent_sha256);assert.equal(t.first_velocity.delta_abs_max,0);}
 else if(t.beta>0)assert.ok(t.first_velocity.delta_abs_max>0);
}
for(const p of data.listening_pairs)assert.ok(data.samples.some(s=>s.id===p.a)&&data.samples.some(s=>s.id===p.b));
const conf=JSON.parse(await readFile(resolve(base,'descriptor_confound_audit.json')));assert.equal(conf.checks.total_real_excerpts,108);assert.equal(conf.checks.all_audio_hashes_match_existing_scores,true);assert.ok(conf.checks.maximum_global_M_reconstruction_error<1e-6);assert.equal(conf.human_labels_count,0);
const sunoSource=sha(await readFile(resolve(base,'runs/2026-11-04-suno-real/features.json'))),controlsSource=sha(await readFile(resolve(base,'runs/2026-11-05-controls/raw-scores.json')));assert.equal(conf.input_sha256.suno_auditory,sunoSource);assert.equal(conf.input_sha256.suno_roughness,sunoSource);assert.deepEqual(conf.input_sha256.controls_auditory,[controlsSource]);assert.deepEqual(conf.input_sha256.controls_roughness,[controlsSource]);
const dest=resolve(root,'platform/site/demos/data/contrast-followup-audit.json');if(process.argv.includes('--check'))assert.equal(await readFile(dest,'utf8'),bytes);else await writeFile(dest,bytes);
console.log('External contrast checked:8 actual renders /16 fixed windows /24 MP3 bindings;2 sham PCM+latent matches;0 human labels.');
