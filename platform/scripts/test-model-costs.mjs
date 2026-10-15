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
    assert.equal(await root.locator('[data-cost="timing"] tr').count(),7);
    assert.match(await root.locator('[data-cost="parameters"]').textContent(),/3,580,352,576/);
    assert.match(await root.locator('[data-cost="parameters"]').textContent(),/13,484,032/);
    assert.match(await root.locator('[data-cost="timing"] [data-cost-model="songbloom"]').textContent(),/原生不支持/);
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
    await root.locator('[data-cost-filter="author"]').click();
    assert.equal(await root.locator('[data-cost="timing"] tr').count(),data.timing.filter(r=>r.kind==='author-reference').length);
    await root.locator('[data-cost-filter="measured"]').click();
    assert.equal(await root.locator('[data-cost="timing"] tr').count(),data.timing.filter(r=>r.kind.startsWith('measured')).length);
    await root.locator('[data-cost-filter="all"]').click();
    const download=await page.request.get(new URL('data/model-costs.json',page.url()).href);
    assert.equal(download.status(),200);
    assert.equal((await download.json()).models.length,6);
    await page.screenshot({path:`${output}/${width}.png`,fullPage:true});
    await root.locator('summary').click();
    assert.match(await root.locator('[data-cost="evidence"]').textContent(),/初始化/);
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
  }
  assert.deepEqual(errors,[]);
  console.log('Six models / exact native and LoRA counts / seven evidence-separated timing rows / filters / JSON / desktop and mobile: PASS');
}finally{await browser.close();}
