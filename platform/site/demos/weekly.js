import {createMilestones} from './chart-milestones.js';
const $=s=>document.querySelector(s),el=(t,text)=>{const e=document.createElement(t);if(text!==undefined)e.textContent=text;return e;};
const names={yue2:'YuE2',diffrhythm2:'DiffRhythm2',muse:'Muse'},rewards={richness:'SP Richness',instrument:'SongBench-Instrument',arrangement:'SongBench-Arrangement',mean7:'SongBench-Mean7'};
const fmt=n=>n.toFixed(3),sign=n=>(n>=0?'+':'')+fmt(n),interval=e=>`${sign(e.delta)} [${e.ci95.map(sign).join(', ')}]`;
const checked=value=>new Date(value).toLocaleString('zh-CN',{timeZone:'Australia/Perth',hour12:false})+' AWST';
let data,run,listenRunId,listenStep=270,example=0,listening,listenRuns,inputs,progression;
const rate=r=>r.lr===.0001?'1e-4':'3e-4';
const marks=Object.fromEntries(['reward','kl'].map(k=>[k,createMilestones($(`[data-curve="${k}"]`),270)]));
function progressGallery(){
 const progressVariant='C';
 $('[data-progress="models"]').replaceChildren(...progression.models.map(m=>{
  const article=el('article'),bar=el('div'),links=el('div');bar.className='section-title';links.className='downloads';
  article.dataset.progressModel=m.id;bar.append(el('h3',m.title));
  for(const type of ['png','svg','pdf']){const a=el('a',`${type.toUpperCase()} ↓`);a.href=`./weekly-2026-10-23/${m.variants[progressVariant][type].file}`;a.download='';links.append(a);}
  bar.append(links);article.append(bar);const figure=el('figure'),a=el('a'),img=el('img');figure.className='result-figure';
  const png=m.variants[progressVariant].png;
  img.src=`./weekly-2026-10-23/${png.file}?v=${png.sha256.slice(0,12)}`;img.alt=`${m.title} · SongBench mean · measured validation at steps 0/50/100/270 · ${progressVariant}`;
  a.href=img.src;a.target='_blank';a.rel='noopener';a.append(img);figure.append(a);article.append(figure);return article;
 }));
}
function draw(){if(!run)return;for(const key of ['reward','kl']){const canvas=$(`[data-curve="${key}"]`),width=canvas.getBoundingClientRect().width;if(width<20)continue;const height=180,dpr=Math.min(devicePixelRatio||1,2);canvas.width=width*dpr;canvas.height=height*dpr;const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);const points=run.trace.map((p,i,a)=>({step:p.step,value:a.slice(Math.max(0,i-9),i+1).reduce((n,r)=>n+r[key],0)/Math.min(10,i+1)})),vs=points.map(p=>p.value),lo=Math.min(...vs),hi=Math.max(...vs),pad=Math.max((hi-lo)*.1,.00001),low=key==='kl'?Math.max(0,lo-pad):lo-pad,high=hi+pad,x=s=>48+(width-62)*s/270,y=v=>12+133*(high-v)/(high-low);ctx.font='10px system-ui';ctx.fillStyle='#627066';ctx.strokeStyle='#dde4df';for(let i=0;i<3;i++){const v=low+(high-low)*i/2,py=y(v);ctx.beginPath();ctx.moveTo(48,py);ctx.lineTo(width-14,py);ctx.stroke();ctx.fillText(v.toFixed(key==='kl'?4:3),0,py+3);}for(const s of [0,50,100,270])ctx.fillText(s,Math.min(width-28,x(s)-6),173);const color=key==='kl'?'#3466a6':'#14816d';ctx.strokeStyle=color;ctx.lineWidth=1.8;ctx.beginPath();points.forEach((p,i)=>i?ctx.lineTo(x(p.step),y(p.value)):ctx.moveTo(x(p.step),y(p.value)));ctx.stroke();marks[key].set(run.trace.filter(p=>[1,5,25,50,100,270].includes(p.step)).map(p=>({step:p.step,label:`Step ${p.step} · ${key} ${p[key].toFixed(6)} · raw log`})));marks[key].draw(ctx,width,149,color);}}
function choose(r){
 run=r;for(const b of $('[data-curve="runs"]').children)b.setAttribute('aria-pressed',String(b.dataset.id===r.id));
 const objective=r.reward==='mean7'?'R = 七维原始分均值（0–10，不除10、不减guard）'
  :r.reward==='richness'?'R = SP Richness − P'
  :`R = ${r.reward==='instrument'?'Instrumental':'Arrangement'} / 10 − P（不是原始0–10分）`;
 $('[data-curve="note"]').textContent=`${names[r.model]} · ${rewards[r.reward]} · ${objective}。P为削波/RMS惩罚；技术无效音频R=0。曲线不是advantage，KL另计：${r.kl_unit}${r.status?' · '+r.status+'；训练七维已有绑定，固定验证尚无结果。':''}`;draw();
}
function audio(){
 const chosen=listenRuns.find(r=>r.id===listenRunId),source=inputs.find(r=>r.index===example),container=$('[data-listen="audio"]');
 $('[data-listen="text"]').textContent=source.style;$('[data-listen="lyrics"]').textContent=source.lyrics;
 $('[data-listen="scope"]').textContent=`${names[chosen.model]} · ${rewards[chosen.reward]} · LR${rate(chosen)}。${chosen.model==='muse'?'原生最多10.24s codec预算，不是60秒整歌。':'原生60s预算，保留实际输出时长。'}同一固定输入、seed与冻结baseline；不是新推理或拼接。`;
 for(const b of $('[data-listen="models"]').children)b.setAttribute('aria-pressed',String(b.dataset.runId===listenRunId));
 for(const b of $('[data-listen="examples"]').children)b.setAttribute('aria-pressed',String(+b.dataset.index===example));
 for(const b of $('[data-listen="steps"]').children)b.setAttribute('aria-pressed',String(+b.dataset.step===listenStep));
 for(const [i,step] of [0,listenStep].entries()){
  const r=listening.find(r=>r.run_id===listenRunId&&r.step===step&&r.index===example),role=i?'candidate':'baseline',previous=container.querySelector(`[data-role="${role}"]`);
  if(!i&&previous?.dataset.binding===r.source_audio_sha256)continue;
  previous?.querySelector('audio')?.pause();
  const s=el('div');s.dataset.role=role;s.dataset.binding=r.source_audio_sha256;
  const base=listening.find(r=>r.run_id===listenRunId&&r.step===0&&r.index===example);
  s.append(el('h3',`${i?'B · GRPO ·'+step:'A · Frozen baseline ·0'} · Mean7 ${fmt(r.mean7)}${i?' ('+sign(r.mean7-base.mean7)+')':''} · ${r.seconds.toFixed(2)}s`));
  const a=el('audio');a.controls=true;a.preload='metadata';a.src=r.url;
  a.addEventListener('play',()=>{for(const o of container.querySelectorAll('audio'))if(o!==a)o.pause();});s.append(a);
  if(previous)previous.replaceWith(s);else container.append(s);
 }
}
try{data=await (await fetch('./weekly-2026-10-23/summary.json',{cache:'no-store'})).json();const n=data.runs.reduce((s,r)=>s+r.binding.bound_audio_n,0);$('[data-summary="scope"]').textContent=`已核验 ${data.run_count} 个SP/SongBench完整实验 + 3个SongEval LR；${n.toLocaleString()}条音频绑定（含共享baseline，不是独立歌曲数）。CMI固定270 train /30 valid；截至 ${checked(data.updated_at)}，非实时监控。`;
progression=await(await fetch('./weekly-2026-10-23/progression.json',{cache:'no-store'})).json();progressGallery();
$('[data-progress="rows"]').replaceChildren(...progression.models.flatMap(m=>m.series.map(s=>{const tr=el('tr');for(const value of [m.title,s.label,'valid / 30',...s.points.map(p=>p?fmt(p.value):'n/a')])tr.append(el('td',value));return tr;})));
const sb=data.runs.find(r=>r.reward==='mean7'),sp=data.runs.filter(r=>r.reward==='richness');$('[data-summary="claims"]').replaceChildren(...[
`SongEval在YuE2上有验证集增益：3e-4由${fmt(data.songeval[2].effect.baseline)}到${fmt(data.songeval[2].effect.final)}，Δ ${interval(data.songeval[2].effect)}。这是优化指标上的验证增益，LR选择未校正，不是人类听感结论。`,
`SP代理可以提高，但跨指标迁移不稳定：${sp.map(r=>`${names[r.model]}的SP Δ ${sign(r.sp.delta)}、SongBench Mean7 Δ ${interval(r.mean7)}`).join('；')}。`,
`SongBench单项与七维均值不是同一个实验：YuE2的Instrument/Arrangement对Mean7呈正向变化。${data.runs.filter(r=>r.reward==='mean7').map(r=>`${names[r.model]} Mean7 LR${rate({lr:r.experiment.lr})}：${fmt(r.mean7.baseline)}→${fmt(r.mean7.final)}，Δ ${interval(r.mean7)}`).join('；')}。`,
`小数据可用于检验post-training响应：这里只有270个训练条件；尚不能推出“小数据足以稳定提高全曲质量”或MOS质量档的普遍规律。`
].map(t=>el('li',t)));
$('[data-summary="songeval"]').replaceChildren(...data.songeval.map(r=>{const t=el('tr');for(const v of [r.lr.toExponential(),fmt(r.effect.baseline),fmt(r.effect.final),interval(r.effect),`${r.effect.positive}/30`])t.append(el('td',v));return t;}));$('[data-summary="regimes"]').replaceChildren(...data.runs.map(r=>{const t=el('tr');for(const v of [`${names[r.model]} · ${rewards[r.reward]} · LR${rate({lr:r.experiment.lr})}${r.id.endsWith('-v2')?' · v2重跑':''}`,fmt(r.mean7.baseline),fmt(r.mean7.final),interval(r.mean7),`${fmt(r.target.baseline)} → ${fmt(r.target.final)}`])t.append(el('td',v));return t;}));$('[data-summary="failures"]').replaceChildren(...data.failures.map(r=>el('li',`${r.model} · ${r.reward} · ${r.job}：${r.status}。${r.reason}。未完成的评估不填入验证结果。`)));
$('[data-summary="cohorts"]').replaceChildren(...data.cohorts.map(r=>{const tr=el('tr');for(const v of [`${names[r.model]} · Mean7 · LR${rate(r)}`,r.split,r.step,`${r.n}/${r.n}`,`${r.scored_n}/${r.n}`,r.mean7===null?'n/a':fmt(r.mean7),r.invalid_n])tr.append(el('td',v));return tr;}));
$('[data-curve="runs"]').replaceChildren(...data.runs.map(r=>{const b=el('button',`${names[r.model]} · ${rewards[r.reward]} · LR${rate({lr:r.experiment.lr})}`);b.dataset.id=r.id;b.addEventListener('click',()=>choose(r));return b;}));choose(data.runs.find(r=>r.model==='muse'&&r.reward==='mean7')||sb);for(const key of ['reward','kl'])new ResizeObserver(draw).observe($(`[data-curve="${key}"]`));document.addEventListener('toggle',draw,true);
const audioData=await(await fetch('./weekly-2026-10-23/listening.json',{cache:'no-store'})).json();listening=audioData.rows;
listenRuns=[...audioData.runs].sort((a,b)=>['diffrhythm2','muse'].indexOf(a.model)-['diffrhythm2','muse'].indexOf(b.model)||['richness','mean7'].indexOf(a.reward)-['richness','mean7'].indexOf(b.reward)||b.lr-a.lr);
inputs=(await(await fetch('./data/reward-regime-examples.json')).json()).inputs;listenRunId=listenRuns.find(r=>r.model==='muse'&&r.reward==='mean7').id;
$('[data-listen="models"]').replaceChildren(...listenRuns.map(r=>{const b=el('button',`${names[r.model]} · ${rewards[r.reward]} · LR${rate(r)}`);b.dataset.runId=r.id;b.addEventListener('click',()=>{listenRunId=r.id;audio();});return b;}));
$('[data-listen="steps"]').replaceChildren(...[0,50,100,270].map(step=>{const b=el('button',`${step} steps`);b.dataset.step=step;b.addEventListener('click',()=>{listenStep=step;audio();});return b;}));
$('[data-listen="examples"]').replaceChildren(...[0,10,20,1,2].map(i=>{const b=el('button',`valid${i}`);b.dataset.index=i;b.addEventListener('click',()=>{example=i;audio();});return b;}));audio();
}catch(e){$('[data-summary="scope"]').textContent='结果载入失败：'+e.message;console.error(e);}
try{const jobs=await(await fetch('./weekly-2026-10-23/next-experiments.json',{cache:'no-store'})).json();$('[data-jobs="scope"]').textContent=`截至 ${checked(jobs.checked_at)} · ${jobs.protocol}`;$('[data-jobs="rows"]').replaceChildren(...jobs.jobs.map(j=>{const tr=el('tr');for(const v of [j.name,j.resource,j.status,j.progress])tr.append(el('td',v));return tr;}));for(const r of jobs.traces||[]){if(!r.trace.length)continue;const b=el('button',`${names[r.model]} · ${rewards[r.reward]} · LR${rate(r)} · ${r.phase}`);b.dataset.id=r.id;b.addEventListener('click',()=>choose(r));$('[data-curve="runs"]').append(b);}}catch(e){$('[data-jobs="scope"]').textContent='作业快照暂不可用。';console.error(e);}
