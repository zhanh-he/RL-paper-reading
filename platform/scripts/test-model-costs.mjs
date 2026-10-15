import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
const data=JSON.parse(await fs.readFile(new URL('../site/demos/data/model-costs.json',import.meta.url)));
const expected=[
  ['yue2',3580352576,8716288],['muse',612555776,5046272],
  ['diffrhythm2',1136249664,6029312],['heartmula',3938113536,13484032],
  ['songbloom',1872372929,3096576],['acestep15',2393872518,10223616],
];
assert.deepEqual(data.models.map(m=>[m.id,m.parameters,m.lora_parameters]),expected);
for(const file of ['native-hf-final-counts.json','native-heart-counts.json','native-songbloom-counts.json']){
  const receipt=JSON.parse(await fs.readFile(new URL(`../../music-gen/lyrics2song/model-costs/${file}`,import.meta.url)));
  for(const item of Object.values(receipt.components)){
    const count=Object.values(item.parameter_shapes).reduce((sum,shape)=>sum+shape.reduce((a,b)=>a*b,1),0);
    assert.equal(count,item.unique_parameters);
    assert.equal(Object.values(item.lora.targets).reduce((sum,shape)=>sum+8*(shape[0]+shape[1]),0),item.lora.parameters);
  }
}
for(const model of data.models){
  assert.equal(model.lora_rank,8);
  assert.equal(model.lora_alpha,16);
  assert.equal(model.lora_percent,100*model.lora_parameters/model.parameters);
}
for(const arm of data.timing){
  for(const row of arm.rows){
    if(row.actual_audio_seconds && row.kind!=='measured')assert.ok(Math.abs(row.wall_seconds/row.actual_audio_seconds-row.rtf)<1e-9);
    assert.ok(row.kind!=='unsupported'||row.wall_seconds===null);
  }
}
assert.equal(data.timing.find(m=>m.id==='songbloom').rows[1].kind,'unsupported');
assert.equal(data.timing.find(m=>m.id==='acestep15').kind,'unavailable');
assert.equal(data.timing.find(m=>m.id==='acestep15-official').kind,'author-reference');
const aceClaim=data.timing.find(m=>m.id==='acestep15-official');
assert.equal(aceClaim.publish_duration_estimates,false);
assert.equal(aceClaim.median_rtf,null);
assert.ok(Math.abs(aceClaim.claim_rtf-2/240)<1e-12);
assert.ok(aceClaim.rows.every(r=>r.wall_seconds===null&&r.kind==='boundary-unverified'));
assert.equal(data.timing.find(m=>m.id==='diffrhythm2').kind,'measured-rollout-proxy');
assert.equal(data.timing_audit.benchmark_contract.status,'proposed; not launched');
assert.equal(data.timing.find(m=>m.id==='diffrhythm2').n,29);
assert.equal(data.timing.find(m=>m.id==='heartmula').n,69);
const musicgen=data.timing.find(m=>m.id==='musicgen-large-reference');
assert.equal(musicgen.reported_rtf,.4);
assert.equal(musicgen.median_rtf,2.5);
assert.deepEqual(musicgen.rows.map(r=>r.wall_seconds),[150,750]);
for(const id of ['levo2-medium-reference']){
  const arm=data.timing.find(m=>m.id===id);
  assert.equal(arm.max_output_seconds,270);
  assert.equal(arm.rows[1].kind,'unsupported');
  assert.equal(arm.rows[1].wall_seconds,null);
  assert.equal(arm.kind,'author-reference');
}
assert.equal(data.timing.some(m=>m.id==='levo2-large-reference'),false);
assert.equal(data.acceleration.models.length,8);
assert.equal(data.acceleration.projections.length,4);
for(const r of data.acceleration.projections){
  assert.ok(r.ar_fraction>0&&r.ar_fraction<1);
  for(const s of r.scenarios){
    assert.equal(s.estimated_wall_seconds,r.ar_seconds/s.assumed_ar_speedup+r.unchanged_seconds);
    assert.ok(s.estimated_e2e_speedup<s.assumed_ar_speedup);
  }
  assert.ok(r.scenarios[1].estimated_e2e_speedup<r.ar_only_infinite_speedup_bound);
}
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'playwright');
const browser=await chromium.launch({headless:true,channel:'chrome'});
const output=process.env.QA_OUTPUT||'/tmp/model-costs-qa';
await fs.mkdir(output,{recursive:true});
try{
  const page=await browser.newPage();
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  for(const width of [1440,390]){
    await page.setViewportSize({width,height:1000});
    await page.goto(process.env.DEMO_URL||'http://127.0.0.1:4186/demos/#model-costs');
    const root=page.locator('#view-model-costs');
    await root.locator('[data-cost="parameters"] tr').last().waitFor();
    assert.equal(await root.isVisible(),true);
    assert.equal(await root.locator('[data-cost="parameters"] tr').count(),6);
    assert.equal(await root.locator('[data-cost="timing"] tr').count(),9);
    assert.equal(await root.locator('[data-cost="acceleration"] tr').count(),8);
    assert.equal(await root.locator('[data-cost="projections"] tr').count(),4);
    assert.equal(await root.locator('[data-cost="timing"] [data-cost-model="levo2-large-reference"]').count(),0);
    assert.match(await root.locator('[data-acceleration-model="acestep15"]').textContent(),/nano-vLLM/);
    assert.match(await root.locator('[data-acceleration-model="yue2"]').textContent(),/vLLM-Omni/);
    assert.match(await root.locator('[data-cost="parameters"]').textContent(),/3,580,352,576/);
    assert.match(await root.locator('[data-cost="parameters"]').textContent(),/13,484,032/);
    assert.match(await root.locator('[data-cost="timing"] [data-cost-model="songbloom"]').textContent(),/原生不支持/);
    assert.match(await root.locator('[data-cost="timing"] [data-cost-model="acestep15-official"]').textContent(),/撤回外推/);
    assert.doesNotMatch(await root.locator('[data-cost="timing"] [data-cost-model="acestep15-official"]').textContent(),/0分0\.50秒|0分2\.50秒/);
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
    await root.locator('[data-cost-filter="author"]').click();
    assert.equal(await root.locator('[data-cost="timing"] tr').count(),data.timing.filter(r=>r.kind==='author-reference').length);
    await root.locator('[data-cost-filter="measured"]').click();
    assert.equal(await root.locator('[data-cost="timing"] tr').count(),data.timing.filter(r=>r.kind.startsWith('measured')).length);
    await root.locator('[data-cost-filter="all"]').click();
    const href=await root.locator('a[download]').getAttribute('href');
    const download=await page.request.get(new URL(href,page.url()).href);
    assert.equal(download.status(),200);
    assert.equal((await download.json()).models.length,6);
    await page.screenshot({path:`${output}/${width}.png`,fullPage:true});
    await root.locator('summary').first().click();
    assert.match(await root.locator('[data-cost="timing-audit"]').textContent(),/nanovllm|nano-vLLM/);
    assert.match(await root.locator('[data-cost="evidence"]').textContent(),/初始化/);
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
    await root.locator('summary').nth(1).click();
    assert.match(await root.locator('[data-cost="projections"]').textContent(),/条件估算/);
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
  }
  assert.deepEqual(errors,[]);
  console.log('Six parameter audits / nine timing rows / LeVo2-medium selection / eight backend rows / conditional Amdahl scenarios / desktop and mobile: PASS');
}finally{await browser.close();}
