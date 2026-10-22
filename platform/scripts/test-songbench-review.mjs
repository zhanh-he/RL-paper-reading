import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {createHash} from 'node:crypto';
const d=JSON.parse(await fs.readFile(new URL('../site/demos/data/songbench-review.json',import.meta.url)));
assert.equal(d.new_training_started,true);
assert.equal(d.mean7_training_started,true);
assert.equal(d.base_identity.starting_from_grpo,false);
assert.equal(d.base_identity.model_sha256,'c810b2a748579e06e960019f89a9cb98701e4ff545e2c760e8ceaa3343ff1b52');
assert.equal(d.priorities.filter(p=>p[0].startsWith('P0')).length,2);
assert.equal(d.runs.length,8);
assert.equal(d.listening.length,20);
assert.equal(d.runs.reduce((s,r)=>s+r.binding.bound_audio_n,0),5220);
for(const r of d.runs){
  assert.equal(r.updates,270);
  assert.ok(r.validation.every(v=>v.n===30));
  assert.equal(r.trace.length,270);
  assert.ok(r.mean7_delta_ci95[0]<=r.mean7_delta&&r.mean7_delta<=r.mean7_delta_ci95[1]);
  assert.ok(r.validation.every(v=>v.invalid_n===0));
}
for(const a of d.listening){
  const file=new URL('../site/demos/'+a.url.replace(/^\.\//,''),import.meta.url);
  assert.equal(createHash('sha256').update(await fs.readFile(file)).digest('hex'),a.public_mp3_sha256);
  assert.notEqual(a.source_audio_sha256,a.public_mp3_sha256);
}
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'playwright');
const browser=await chromium.launch({headless:true});
try{
  const page=await browser.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));
  const base=process.env.DEMO_URL||'http://127.0.0.1:4186/demos/#lyrics-model-plan';
  for(const width of [1440,390]){
    await page.setViewportSize({width,height:1000});await page.goto(base);
    const root=page.locator('[data-songbench-review]');
    await root.locator('[data-sb="results"] tr').last().waitFor();
    assert.equal(await root.isVisible(),true);
    assert.equal(await root.locator('[data-sb="results"] tr').count(),8);
    assert.match(await root.textContent(),/唯一 reward/);
    assert.match(await root.locator('[data-sb="failures"]').textContent(),/generate_objective/);
    assert.match(await root.locator('[data-sb="execution"]').textContent(),/Muse-SP/);
    assert.match(await root.locator('[data-sb="launch-link"]').getAttribute('href'),/weekly-2026-10-23$/);
    await root.getByText('已回收训练曲线 · 左Reward / 右KL',{exact:true}).evaluate(s=>{s.parentElement.open=true;});
    assert.equal(await root.locator('[data-sb="runs"] button').count(),8);
    await root.locator('[data-sb="runs"] button').last().click();
    assert.match(await root.locator('[data-sb="target-caption"]').textContent(),/无guard/);
    for(const canvas of await root.locator('canvas').all()){
      assert.ok(await canvas.evaluate(c=>c.width>100&&c.getContext('2d').getImageData(0,0,c.width,c.height).data.some(v=>v)));
      const bounds=await canvas.boundingBox();await page.mouse.move(bounds.x+40,bounds.y+70);
    }
    await root.getByText('Diff主干历史试听 · 五个固定example / 20条已有MP3',{exact:true}).evaluate(s=>{s.parentElement.open=true;});
    assert.equal(await root.locator('audio').count(),4);
    const source=await root.locator('audio').first().getAttribute('src');
    await root.locator('[data-sb="runs"] button').first().click();
    assert.equal(await root.locator('audio').first().getAttribute('src'),source);
    for(const i of [0,10,20,1,2]){
      await root.locator(`[data-index="${i}"]`).click();
      assert.match(await root.locator('audio').first().getAttribute('src'),new RegExp(`baseline-${String(i).padStart(3,'0')}\\.mp3$`));
      assert.ok((await root.locator('[data-sb="lyrics"]').textContent()).length>50);
    }
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
    await root.locator('audio').first().evaluate(a=>new Promise((resolve,reject)=>{if(a.readyState>=1)return resolve();a.addEventListener('loadedmetadata',resolve,{once:true});a.addEventListener('error',reject,{once:true});}));
    assert.ok(await root.locator('audio').first().evaluate(a=>a.duration>59&&a.duration<61));
    await page.evaluate(()=>scrollTo(0,0));
    await page.screenshot({path:`/tmp/songbench-review-${width}.png`,fullPage:true});
  }
  for(const a of d.listening){const response=await page.request.get(new URL(a.url,page.url()).href);assert.equal(response.status(),200);assert.equal(createHash('sha256').update(await response.body()).digest('hex'),a.public_mp3_sha256);}
  await page.goto(new URL('#reward-regimes',base).href);
  const regime=page.locator('#view-reward-regimes');
  await regime.locator('[data-model="muse"]').click();
  await regime.getByRole('button',{name:'Muse-SP-Richness-3e-4',exact:true}).click();
  assert.match(await regime.locator('[data-regime="status"]').textContent(),/已核验/);
  assert.match(await regime.locator('[data-regime="train-protocol"]').textContent(),/原始非GRPO Muse/);
  assert.match(await regime.locator('[data-regime="train-protocol"]').textContent(),/已核验270步/);
  for(const [name,id] of [['Instrument','181718185'],['Arrangement','181718187']]){
    await regime.getByRole('button',{name:`Muse-SongBench-${name}-3e-4`,exact:true}).click();
    assert.match(await regime.locator('[data-regime="status"]').textContent(),/启动失败/);
    assert.ok((await regime.locator('[data-regime="train-protocol"]').textContent()).includes(id));
  }
  assert.deepEqual(errors,[]);
  console.log('PASS: eight complete arms, Muse-SP completion, explicit startup failures,20 MP3 hashes, Reward/KL canvas, desktop/mobile.');
}finally{await browser.close();}
