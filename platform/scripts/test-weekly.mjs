import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {createHash} from 'node:crypto';
const root=new URL('../site/demos/weekly-2026-10-23/',import.meta.url);
const d=JSON.parse(await fs.readFile(new URL('summary.json',root)));
assert.equal(await fs.readFile(new URL('summary.json',root),'utf8'),await fs.readFile(new URL('../../music-gen/reward-regime-study/weekly-2026-10-23/summary.json',import.meta.url),'utf8'));
assert.equal(d.runs.length,8);assert.equal(d.songeval.length,3);assert.equal(d.audited_audio_bindings,5220);
const mean=a=>a.reduce((s,v)=>s+v,0)/a.length;
for(const r of d.runs){
  const a=r.validation[0],b=r.validation.at(-1);assert.equal(r.trace.length,270);
  assert.ok(r.validation.every(v=>v.n===30&&v.rows.length===30));
  assert.ok(Math.abs(mean(b.rows.map(x=>x.mean7))-mean(a.rows.map(x=>x.mean7))-r.mean7.delta)<1e-12);
  for(const e of [r.mean7,r.target,...Object.values(r.heads)]){
    assert.equal(e.n,30);assert.equal(e.paired_deltas.length,30);assert.ok(Math.abs(mean(e.paired_deltas)-e.delta)<1e-12);
    assert.ok(e.ci95[0]<=e.delta&&e.delta<=e.ci95[1]);
  }
}
assert.equal(d.failures.length,3);assert.ok(d.failures.every(r=>r.status.startsWith('failed')));
assert.equal(d.figures.length,4);
for(const f of d.figures)for(const v of ['A','B','C'])for(const type of ['png','pdf','svg']){
  const file=f.variants[v][type],bytes=await fs.readFile(new URL(file.file,root));
  assert.equal(createHash('sha256').update(bytes).digest('hex'),file.sha256);
  if(type==='png'){assert.equal(bytes.readUInt32BE(16),2400);assert.equal(bytes.readUInt32BE(20),1350);}
}
const progressionText=await fs.readFile(new URL('progression.json',root),'utf8'),progression=JSON.parse(progressionText);
assert.equal(progressionText,await fs.readFile(new URL('../../music-gen/reward-regime-study/weekly-2026-10-23/progression.json',import.meta.url),'utf8'));
assert.equal(progression.source_sha256,createHash('sha256').update(await fs.readFile(new URL('summary.json',root))).digest('hex'));
assert.deepEqual(progression.steps,[0,50,100,270]);assert.equal(progression.metric,'SongBench_mean7');
assert.deepEqual(progression.models.map(m=>m.id),['diffrhythm2','yue2','muse']);
const heads=['Vocal','Instrumental','Melody','Structure','Arrangement','Mixing','Musicality'];
const crossBytes=await fs.readFile(new URL('songeval-cross.json',root)),cross=JSON.parse(crossBytes);
assert.equal(cross.status,'verified-complete');assert.equal(cross.binding.bound_audio_n,120);
assert.equal(createHash('sha256').update(crossBytes).digest('hex'),progression.extra_source_sha256['songeval-cross.json']);
for(const m of progression.models){
 assert.equal(m.series.length,m.id==='muse'?1:2);
 assert.deepEqual(Object.keys(m.variants),['C']);
 for(const s of m.series){
  assert.equal(s.split,'valid');assert.equal(s.lr,.0003);
  const run=d.runs.find(r=>r.id===s.id)||(s.id===cross.id?cross:null);assert.ok(run);
  if(s.label.includes('SongEval'))assert.equal(run.id,cross.id);
  for(const [i,p] of s.points.entries()){
   const receipt=run.validation.find(v=>v.step===progression.steps[i]);
   if(!receipt){assert.equal(p,null);continue;}
   assert.equal(p.step,receipt.step);assert.equal(p.n,30);assert.equal(p.receipt_sha256,receipt.receipt_sha256);
   const independentlyComputed=mean(receipt.rows.map(r=>mean(heads.map(h=>r.songbench[h]))));
   assert.ok(Math.abs(independentlyComputed-p.value)<1e-12);assert.ok(p.value>=0&&p.value<=10);
  }
  if(s.baseline_kind==='regime')assert.equal(s.points[0].value,m.baseline);
  else {assert.equal(s.baseline_kind,'songeval-archive');assert.notEqual(s.points[0].value,m.baseline);}
 }
 for(const v of ['C'])for(const type of ['png','pdf','svg']){
  const file=m.variants[v][type],bytes=await fs.readFile(new URL(file.file,root));
  assert.equal(createHash('sha256').update(bytes).digest('hex'),file.sha256);
  if(type==='png'){assert.equal(bytes.readUInt32BE(16),2400);assert.equal(bytes.readUInt32BE(20),1350);}
 }
}
const jobsText=await fs.readFile(new URL('next-experiments.json',root),'utf8'),jobs=JSON.parse(jobsText);
assert.equal(jobsText,await fs.readFile(new URL('../../music-gen/reward-regime-study/weekly-2026-10-23/next-experiments.json',import.meta.url),'utf8'));
assert.equal(jobs.jobs.length,7);assert.equal(jobs.traces.length,1);
for(const r of jobs.traces){assert.ok(r.status.includes('不是完整验证'));for(const [i,p] of r.trace.entries()){assert.equal(p.step,i+1);assert.ok(Number.isFinite(p.reward)&&Number.isFinite(p.kl));}}
const progressionCSV=await fs.readFile(new URL('progression.csv',root),'utf8');
assert.equal(progressionCSV,await fs.readFile(new URL('../../music-gen/reward-regime-study/weekly-2026-10-23/progression.csv',import.meta.url),'utf8'));
for(const line of progressionCSV.trim().split('\n').slice(1)){
 const [model,id,reward,lr,split,step,value,n,status,sha]=line.split(',');
 if(split!=='valid'){assert.equal(value,'');assert.equal(n,'');assert.equal(status,'unavailable');assert.equal(sha,'');}
}
const listening=JSON.parse(await fs.readFile(new URL('listening.json',root))).rows;
assert.equal(listening.length,20);
for(const r of listening){
  assert.match(r.url,/\.mp3$/);const bytes=await fs.readFile(new URL('../'+r.url.replace(/^\.\//,''),root));
  assert.equal(createHash('sha256').update(bytes).digest('hex'),r.public_mp3_sha256);
}
if(process.argv.includes('--data-only')){console.log('Weekly: complete source runs +120 real cross-scores;45 figure hashes and20 MP3 hashes verified; focused plots only, missing splits blank.');process.exit(0);}
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'playwright');
const browser=await chromium.launch({headless:true});
try{
 const page=await browser.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));
 for(const width of [1440,390]){
  await page.setViewportSize({width,height:1000});await page.goto(process.env.WEEKLY_URL||'http://127.0.0.1:4186/demos/weekly-2026-10-23.html');
  await page.locator('[data-summary="regimes"] tr').last().waitFor({state:'attached'});await page.locator('[data-listen="audio"] audio').last().waitFor({state:'attached'});
  await page.locator('[data-jobs="rows"] tr').last().waitFor();
  assert.equal(await page.locator('[data-summary="regimes"] tr').count(),8);
  assert.equal(await page.locator('[data-summary="songeval"] tr').count(),3);
  assert.equal(await page.locator('[data-summary="failures"] li').count(),3);
  assert.equal(await page.locator('[data-progress="rows"] tr').count(),5);
  assert.equal(await page.locator('[data-jobs="rows"] tr').count(),7);
  assert.equal(await page.locator('[data-progress-variant], [data-gallery]').count(),0);
  assert.equal(await page.locator('[data-progress="models"] article').count(),3);
  for(const v of ['C']){
   for(const m of progression.models){
    const article=page.locator(`[data-progress-model="${m.id}"]`),img=article.locator('img');
    await img.evaluate(i=>i.decode());assert.equal(await img.evaluate(i=>i.naturalWidth),2400);
    assert.ok((await img.getAttribute('src')).endsWith(`progression-${m.id}-${v}.png`));
    for(const type of ['png','svg','pdf'])assert.ok(await article.locator(`a[download][href$="-${v}.${type}"]`).count());
   }
  }
  await page.locator('[data-progress-model="diffrhythm2"] img').evaluate(i=>i.decode());
  await page.locator('[data-progress-model="diffrhythm2"]').screenshot({path:`/tmp/weekly-progression-${width}.png`});
  await page.getByText('训练过程 · 左 Reward / 右 KL',{exact:true}).evaluate(s=>s.parentElement.open=true);
  for(const b of await page.locator('[data-curve="runs"] button').all())await b.click();
  for(const canvas of await page.locator('canvas').all())assert.ok(await canvas.evaluate(c=>c.width>100&&c.getContext('2d').getImageData(0,0,c.width,c.height).data.some(v=>v)));
  await page.locator('#listening > details').evaluate(e=>e.open=true);
  for(const model of ['diffrhythm2','muse']){
   await page.locator(`[data-listen="models"] [data-model="${model}"]`).click();
   const baseline=await page.locator('audio').first().getAttribute('src');
   await page.locator('[data-curve="runs"] button').first().click();assert.equal(await page.locator('audio').first().getAttribute('src'),baseline);
   for(const i of [0,10,20,1,2]){await page.locator(`[data-listen="examples"] [data-index="${i}"]`).click();
    const audio=page.locator('audio').first();await audio.evaluate(a=>new Promise((resolve,reject)=>{if(a.readyState>=1)return resolve();a.addEventListener('loadedmetadata',resolve,{once:true});a.addEventListener('error',reject,{once:true});}));
    const duration=await audio.evaluate(a=>a.duration),expected=listening.find(r=>r.model===model&&r.step===0&&r.index===i).seconds;
    assert.ok(Math.abs(duration-expected)<.08,`${model}/${i}: playback ${duration}, original ${expected}`);
   }
  }
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
  await page.locator('#next-experiments').screenshot({path:`/tmp/weekly-jobs-${width}.png`});
 }
 assert.deepEqual(errors,[]);console.log('PASS: source means,45 figure hashes,20 MP3 hashes,three focused-only plots, real SongEval own baseline, job states, curves, audio, desktop/mobile.');
}finally{await browser.close();}
