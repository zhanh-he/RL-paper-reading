import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const output = process.env.QA_OUTPUT || '/tmp/reward-regime-qa';
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true, args:['--autoplay-policy=no-user-gesture-required'] });
try {
  const page = await browser.newPage(); const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  for (const width of [1440, 390]) {
    await page.setViewportSize({ width, height: 1000 });
    await page.goto(process.env.DEMO_URL || 'http://127.0.0.1:4179/demos/#reward-regimes');
    const root = page.locator('#view-reward-regimes');
    await root.locator('[data-regime="models"] button').first().waitFor();
    assert.equal(await root.isVisible(), true);
    assert.deepEqual(await root.locator('[data-regime="models"] button').allTextContents(),
      ['YuE2', 'HeartMuLa', 'DiffRhythm2', 'SongBloom', 'Muse', 'ACE-Step 1.5', 'Private baseline']);
    for (const button of await root.locator('[data-regime="models"] button').all()) {
      await button.click();
      assert.equal(await button.getAttribute('aria-pressed'), 'true');
      assert.equal(await root.locator('[data-regime="arms"] tr').count(), 3);
      assert.match(await root.locator('[data-regime="model-note"]').textContent(), /MOS 未测/);
      assert.match(await root.locator('[data-regime="backbone-status"]').textContent(), /LR3e-4.*β0.05/);
      const name = await button.textContent();
      if (['HeartMuLa', 'DiffRhythm2', 'SongBloom', 'Private baseline', 'ACE-Step 1.5'].includes(name)) {
        assert.equal(await root.locator('[data-regime="plots"]').isVisible(), ['HeartMuLa','DiffRhythm2','SongBloom'].includes(name));
        if (['HeartMuLa','DiffRhythm2','SongBloom'].includes(name)) {
          assert.match(await root.locator('[data-regime="model-note"]').textContent(), /Head-only.*待主干LoRA替代/);
          assert.match(await root.locator('[data-regime="status"]').textContent(), /待替代/);
        }
        assert.equal(await root.locator('audio').count(), name === 'DiffRhythm2' ? 2 : 0);
        assert.match(await root.locator('[data-regime="validation"]').textContent(), /n\/a/);
        assert.equal(await root.locator('[data-regime-listen="baseline"]').isDisabled(), name !== 'DiffRhythm2');
        assert.equal(await root.locator('[data-regime-listen="candidate"]').isDisabled(), name !== 'DiffRhythm2');
        if (name === 'DiffRhythm2') {
          assert.match(await root.locator('[data-regime="replay-note"]').textContent(), /旧head-only消融/);
          const audio=root.locator('[data-regime="media"] audio');
          const baseline=await audio.first().getAttribute('src');
          await root.locator('[data-regime="step-controls"]').getByRole('button',{name:'0 steps',exact:true}).click();
          assert.equal(await audio.last().getAttribute('src'),baseline);
          for (const step of [50,100]) {
            await root.locator('[data-regime="step-controls"]').getByRole('button',{name:`${step} steps`,exact:true}).click();
            assert.equal(await audio.first().getAttribute('src'),baseline);
          }
          await root.locator('[data-regime="reward-controls"] button').nth(1).click();
          assert.equal(await root.locator('[data-regime="media"] audio').count(),1);
          assert.equal(await root.locator('[data-regime="media"] audio').getAttribute('src'),baseline);
          await root.locator('[data-regime="reward-controls"] button').first().click();
        }
      }
    }
    const data = await page.evaluate(async () => (await fetch('./data/reward-regime-protocol.json')).json());
    assert.equal(data.training_authorized, true); assert.equal(data.runs.length, 15);
    assert.deepEqual(data.design.lr_grid,['3e-4']);
    assert.equal(data.rewards[2].name, 'SongBench-Arrangement');
    assert.equal(await root.locator('audio').count(), 0);
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth+1));
    await root.locator('[data-regime="models"] button').first().click();
    assert.equal(await root.locator('[data-regime="example-controls"] button').count(),5);
    const originalLyrics = await root.locator('[data-regime="lyrics-content"]').textContent();
    assert.ok(originalLyrics.length > 50);
    for (const [i,count] of [[3,2],[4,5]]) {
      await root.locator('[data-regime="example-controls"] button').nth(i).click();
      assert.match(await root.locator('[data-regime="text-content"]').textContent(),new RegExp(`exactly ${count} accompaniment`));
      assert.equal(await root.locator('[data-regime="lyrics-content"]').textContent(),originalLyrics);
      assert.match(await root.locator('[data-regime="seed"]').textContent(),/5101/);
      assert.equal(await root.locator('audio').count(),2);
      await page.waitForFunction(()=>Array.from(document.querySelectorAll('[data-regime="media"] audio')).every(a=>Number.isFinite(a.duration)&&a.duration>=30));
      assert.match(await root.locator('[data-regime="input-note"]').textContent(),/不计入全30条验证/);
      assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
    }
    await root.locator('[data-regime="example-controls"] button').first().click();
    const canvas=root.locator('[data-regime="reward-chart"]');
    await canvas.waitFor({state:'visible'});
    assert.match(await canvas.getAttribute('aria-description'), /Step 50/);
    assert.match(await canvas.getAttribute('aria-description'), /Step 270/);
    assert.match(await root.locator('[data-regime="arms"]').textContent(), /30\/30/);
    const pixels=await canvas.evaluate((c)=>{
      const values=c.getContext('2d').getImageData(0,0,c.width,c.height).data;
      return values.filter((v,i)=>i%4===3 && v>0).length;
    });assert.ok(pixels>500);
    await canvas.scrollIntoViewIfNeeded();
    const bounds=await canvas.boundingBox();
    await page.mouse.move(bounds.x+39+(bounds.width-51)*50/270,bounds.y+70);
    await root.locator('.training-chart-tooltip:not([hidden])').first().waitFor();
    assert.match(await root.locator('.training-chart-tooltip:not([hidden])').first().textContent(),/Step 50/);
    await root.locator('[data-regime="reward-controls"] button').nth(1).click();
    assert.equal(await root.locator('[data-regime="plots"]').isVisible(),true);
    assert.match(await root.locator('[data-regime="reward-caption"]').textContent(), /Instrumental \/ 10 − P/);
    assert.match(await root.locator('[data-regime="reward-definition"]').textContent(), /固定除10.*削波项最高0.5 \+ RMS项最高0.1/);
    const trace = await page.evaluate(async () => (await (await fetch('./data/reward-regime-progress.json')).json()).runs.find(r=>r.model==='yue2'&&r.reward==='instrument').trace);
    const domain = await canvas.evaluate(c=>({low:Number(c.dataset.axisMin),high:Number(c.dataset.axisMax)}));
    const values = trace.map(p=>p.reward),low=Math.min(...values),high=Math.max(...values),padding=Math.max(.12*(high-low),.000005);
    assert.ok(Math.abs(domain.low-(low-padding))<1e-12);
    assert.ok(Math.abs(domain.high-(high+padding))<1e-12);
    assert.ok(domain.low>0, 'positive SongBench rewards must not be compressed by a forced zero origin');
    assert.match(await root.locator('[data-regime="trace-note"]').textContent(), /保留全部原值范围.*不是组内标准化/);
    await root.locator('[data-regime="training"]').screenshot({path:`${output}/instrument-training-${width}.png`});
    await root.locator('[data-regime="reward-controls"] button').nth(2).click();
    assert.equal(await root.locator('[data-regime="plots"]').isVisible(),true);
    assert.match(await root.locator('[data-regime="reward-caption"]').textContent(), /Arrangement \/ 10 − P/);
    assert.match(await root.locator('[data-regime="status"]').textContent(),/270步/);
    await root.locator('[data-regime="reward-controls"] button').first().click();
    const media=root.locator('[data-regime="media"]');
    assert.equal(await media.locator('audio').count(),2);
    const baseline=await media.locator('audio').first().getAttribute('src');
    const baselineSha=await media.locator('audio').first().getAttribute('data-source-sha');
    await media.locator('[data-role="baseline"]').evaluate((e)=>{e.dataset.preserved='yes';});
    for(const step of ['50 steps','100 steps','270 steps']) {
      await root.locator('[data-regime="step-controls"]').getByRole('button',{name:step,exact:true}).click();
      assert.equal(await media.locator('audio').first().getAttribute('src'),baseline);
      assert.equal(await media.locator('audio').first().getAttribute('data-source-sha'),baselineSha);
      assert.equal(await media.locator('[data-role="baseline"]').getAttribute('data-preserved'),'yes');
    }
    await page.waitForFunction(()=>Array.from(document.querySelectorAll('[data-regime="media"] audio')).every(a=>Number.isFinite(a.duration)&&a.duration>=30));
    for(const audio of await media.locator('audio').all()) {
      assert.match(await audio.getAttribute('src'),/\.mp3$/);
      await audio.evaluate(a=>a.play());await audio.evaluate(a=>a.pause());
    }
    await root.locator('[data-regime-listen="baseline"]').click();
    await page.waitForFunction(()=>!document.querySelector('[data-regime="media"] [data-role="baseline"] audio').paused);
    await media.locator('[data-role="baseline"] audio').evaluate(a=>{a.currentTime=4;});
    await root.locator('[data-regime-listen="candidate"]').click();
    await page.waitForFunction(()=>{
      const a=document.querySelector('[data-regime="media"] [data-role="baseline"] audio');
      const b=document.querySelector('[data-regime="media"] [data-role="candidate"] audio');
      return a.paused&&!b.paused&&b.currentTime>=4;
    });
    await media.locator('[data-role="candidate"] audio').evaluate(a=>a.pause());
    await root.locator('[data-regime="reward-controls"] button').nth(1).click();
    assert.equal(await media.locator('[data-role="baseline"]').getAttribute('data-preserved'),'yes');
    assert.equal(await media.locator('[data-role="baseline"] audio').getAttribute('src'),baseline);
    await root.locator('[data-regime="reward-controls"] button').first().click();
    await media.scrollIntoViewIfNeeded();
    await page.waitForFunction(()=>Array.from(document.querySelectorAll('[data-regime="media"] img')).every(i=>i.complete&&i.naturalWidth>0));
    await media.screenshot({path:`${output}/media-${width}.png`});
    await root.locator('[data-mode="bad"]').click();
    assert.equal(await root.locator('[data-regime="example-controls"] button').count(),2);
    assert.match(await root.locator('[data-regime="replay-note"]').textContent(),/短验证音频/);
    assert.equal(await media.locator('audio').count(),2);
    await root.locator('[data-regime="reward-controls"] button').nth(1).click();
    assert.equal(await root.locator('[data-regime="example-controls"] button').count(),0);
    await root.locator('[data-regime="reward-controls"] button').first().click();
    await root.locator('[data-mode="long"]').click();
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
    const folds=root.locator('details:not([data-regime="training"]):not(.regime-lyrics)');
    assert.ok((await folds.evaluateAll((items)=>items.filter(e=>e.open).length))===0);
    await page.evaluate(()=>window.scrollTo(0,0));
    await page.screenshot({ path: `${output}/${width}.png`, fullPage: true });
    await page.screenshot({ path: `${output}/viewport-${width}.png` });
  }
  assert.deepEqual(errors, []);
  console.log('SP/AI: seven-model roster, Lyrics2song-style controls/foldouts, Reward/KL, immutable baseline across steps and rewards, synchronized playable MP3, badcases, pending isolation, desktop/mobile/no-overflow passed');
} finally { await browser.close(); }
