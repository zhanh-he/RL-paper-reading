import {createMilestones} from './chart-milestones.js';
const root=document.querySelector('[data-songbench-review]');
const $=key=>root.querySelector(`[data-sb="${key}"]`);
const el=(tag,text)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;return e;};
const fmt=n=>n.toFixed(3),signed=n=>(n>=0?'+':'')+fmt(n);
const names={richness:'SP Richness',instrument:'SongBench-Instrument',arrangement:'SongBench-Arrangement'};
const charts=['reward','kl'].map(key=>({key,canvas:$(`${key}-chart`),marks:createMilestones($(`${key}-chart`),270)}));
let data,active,index=0,inputs;
function draw(){
  if(!active)return;
  for(const {key,canvas,marks} of charts){
    const width=canvas.getBoundingClientRect().width;if(width<30)continue;
    const dpr=Math.min(devicePixelRatio||1,2),height=154;
    canvas.width=Math.round(width*dpr);canvas.height=height*dpr;
    const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);
    const points=active.trace.map((row,i,all)=>({step:row.step,value:all.slice(Math.max(0,i-9),i+1).reduce((s,r)=>s+r[key],0)/Math.min(i+1,10)}));
    const values=points.map(p=>p.value),lo=Math.min(...values),hi=Math.max(...values),pad=Math.max((hi-lo)*.1,.00001);
    const low=Math.max(0,lo-pad),high=hi+pad;
    const x=step=>39+(width-51)*step/270,y=value=>12+118*(high-value)/(high-low);
    ctx.font='10px system-ui';ctx.fillStyle='#68726d';ctx.strokeStyle='#dbe2dd';
    for(let i=0;i<3;i++){const value=low+(high-low)*i/2,py=y(value);ctx.beginPath();ctx.moveTo(39,py);ctx.lineTo(width-12,py);ctx.stroke();ctx.fillText(value.toFixed(key==='kl'?4:3),0,py+3);}
    for(const step of [0,50,100,270])ctx.fillText(String(step),Math.min(width-26,x(step)-7),150);
    const color=key==='kl'?'#315fa8':'#08745d';ctx.strokeStyle=color;ctx.lineWidth=1.8;ctx.beginPath();
    points.forEach((p,i)=>i?ctx.lineTo(x(p.step),y(p.value)):ctx.moveTo(x(p.step),y(p.value)));ctx.stroke();
    marks.set(active.trace.filter(p=>[1,5,25,50,100,270].includes(p.step)).map(p=>({step:p.step,label:`Step ${p.step} · ${key} ${p[key].toFixed(6)} · 原日志`})));marks.draw(ctx,width,132,color);
  }
}
function choose(run){
  active=run;$('curve-note').textContent=`${run.model} · ${names[run.reward]} - guard · LR3e-4 /beta${run.experiment.beta} · KL: ${run.kl_unit}`;
  for(const b of $('runs').children)b.setAttribute('aria-pressed',String(b.dataset.run===run.id));
  draw();
}
function audio(){
  for(const old of $('audio').querySelectorAll('audio'))old.pause();
  const row=inputs.find(r=>r.index===index);$('text').textContent=row.style;$('lyrics').textContent=row.lyrics;
  for(const b of $('examples').children)b.setAttribute('aria-pressed',String(+b.dataset.index===index));
  $('audio').replaceChildren(...['baseline','richness','instrument','arrangement'].map((arm,i)=>{
    const receipt=data.listening.find(r=>r.arm===arm&&r.index===index);
    const section=el('section');section.append(el('h4',`${'ABCD'[i]} · ${arm==='baseline'?'Baseline':names[arm]+' ·270'}`),el('p',`Mean7 ${fmt(receipt.mean7)} · ${receipt.seconds.toFixed(1)}s`));
    const player=el('audio');player.controls=true;player.preload='metadata';player.src=receipt.url;player.setAttribute('aria-label',`${arm} valid${index}`);player.addEventListener('play',()=>{for(const other of $('audio').querySelectorAll('audio'))if(other!==player)other.pause();});section.append(player);return section;
  }));
}
try{
  const response=await fetch('./data/songbench-review.json?v=20261020');if(!response.ok)throw Error(`HTTP ${response.status}`);data=await response.json();
  const inputResponse=await fetch('./data/reward-regime-examples.json');if(!inputResponse.ok)throw Error(`Inputs HTTP ${inputResponse.status}`);inputs=(await inputResponse.json()).inputs;
  $('priorities').replaceChildren(...data.priorities.map(row=>{const tr=el('tr');for(const text of row)tr.append(el('td',text));return tr;}));
  $('execution').textContent=data.execution;$('launch-link').href=data.launch_url;
  $('results').replaceChildren(...data.runs.map(run=>{const first=run.validation[0],last=run.validation.at(-1),tr=el('tr');for(const value of [
    `${run.model} · ${names[run.reward]} - guard`,`${run.updates}步 · 0/50/100/270各30/30`,
    `${fmt(first.means.SongBench_mean7)} → ${fmt(last.means.SongBench_mean7)}`,
    `${signed(run.mean7_delta)} [${run.mean7_delta_ci95.map(signed).join(', ')}]`,
  ])tr.append(el('td',value));return tr;}));
  $('failures').textContent=data.failure;$('audit-link').href=data.audit_url;
  $('runs').replaceChildren(...data.runs.map(run=>{const b=el('button',`${run.model} · ${names[run.reward]}`);b.type='button';b.dataset.run=run.id;b.addEventListener('click',()=>choose(run));return b;}));
  $('examples').replaceChildren(...[0,10,20,1,2].map(i=>{const b=el('button',`valid${i}`);b.type='button';b.dataset.index=i;b.addEventListener('click',()=>{index=i;audio();});return b;}));
  choose(data.runs.find(r=>r.model==='diffrhythm2'&&r.reward==='arrangement'));audio();
  for(const chart of charts)new ResizeObserver(draw).observe(chart.canvas);
  root.addEventListener('toggle',draw,true);
}catch(error){$('failures').textContent=`回收结果加载失败：${error.message}`;}
