import { createMilestones } from './chart-milestones.js';
const root = document.querySelector('#view-reward-regimes');
const $ = (id) => root.querySelector(`[data-regime="${id}"]`);
const row = (values, heading = false) => {
  const tr = document.createElement('tr');
  for (const value of values) {
    const cell = document.createElement(heading ? 'th' : 'td');
    if (value instanceof Node) cell.append(value); else cell.textContent = value;
    tr.append(cell);
  }
  return tr;
};
let protocol, progress, selected = 'yue2', target = 'richness';
const rewardKey = (id) => ({ songbench: 'instrument', 'songbench-arrangement': 'arrangement' })[id] || id;
const status = { running: '运行中', 'local-sequential-queue': '本机连续队列', 'pbs-queued': 'PBS 排队（未起跑）' };
const markers = new Map(['reward', 'kl'].map((key) => [key, createMilestones($(`${key}-chart`), 270)]));

function chart(key, points) {
  const canvas = $(`${key}-chart`), width = canvas.clientWidth, height = canvas.clientHeight;
  if (!width || !height) return;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(width*dpr); canvas.height = Math.round(height*dpr);
  const ctx = canvas.getContext('2d'); ctx.scale(dpr,dpr);
  const values = points.map((p) => p[key]);
  let low = Math.min(0,...values), high = Math.max(...values);
  if (high-low < .00001) high = low+.00001;
  const padding = .12*(high-low); high+=padding; if (low<0) low-=padding;
  const axisY=height-30, color=key==='reward'?'#177e70':'#ba514d';
  const x=(step)=>39+(width-51)*step/270;
  const y=(value)=>12+(axisY-12)*(high-value)/(high-low);
  ctx.font='11px system-ui'; ctx.lineWidth=1;
  for (let i=0;i<4;i++) {
    const value=low+(high-low)*i/3, yy=y(value);
    ctx.strokeStyle='#dde4df'; ctx.beginPath();ctx.moveTo(39,yy);ctx.lineTo(width-12,yy);ctx.stroke();
    ctx.fillStyle='#59615c';ctx.textAlign='right';ctx.fillText(value.toFixed(key==='kl'?4:2),35,yy+4);
  }
  ctx.textAlign='center';ctx.fillStyle='#59615c';
  for (const step of [0,50,100,200,270]) ctx.fillText(String(step),x(step),height-8);
  ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();
  points.forEach((p,i)=>{
    const window=points.slice(Math.max(0,i-9),i+1);
    const mean=window.reduce((n,q)=>n+q[key],0)/window.length;
    if(i)ctx.lineTo(x(p.step),y(mean));else ctx.moveTo(x(p.step),y(mean));
  });ctx.stroke();
  const milestones=points.filter((p)=>[1,5,25,50,100,270].includes(p.step));
  markers.get(key).set(milestones.map((p)=>({step:p.step,label:`Step ${p.step}\n${key==='kl'?'KL':'Reward'} ${p[key].toFixed(6)}（日志原值）`})));
  markers.get(key).draw(ctx,width,axisY,color);
}

function draw() {
  if (!progress) return;
  const data=progress.runs.find((r)=>r.model===selected && r.reward===target);
  const trace=data?.trace || [];
  $('plots').hidden=!trace.length;
  $('trace-note').textContent=trace.length
    ? `${selected==='yue2'?'YuE2':'Muse'}-${target}-3e-4 · 已核验 ${data.steps}/270 步。曲线为10步滑动均值，悬停打点显示日志原值。Reward 不包含 KL loss；KL 是同一 rollout 上的在线参考估计，不是精确全策略 KL，也不是测试分数。`
    : '尚无已核验训练日志。排队状态不代表已训练，不显示替代曲线。';
  if(trace.length) { chart('reward',trace);chart('kl',trace); }
}

function render() {
  for (const b of $('models').querySelectorAll('button')) b.setAttribute('aria-pressed', String(b.dataset.model === selected));
  for (const b of $('reward-controls').querySelectorAll('button')) b.setAttribute('aria-pressed',String(b.dataset.reward===target));
  const model = protocol.models.find((m) => m.id === selected);
  $('model-note').textContent = `${model.name} · ${model.role} · ${model.status} · MOS 未测。${model.note}`;
  $('arms').replaceChildren(...protocol.rewards.map((r)=>{
    const reward=rewardKey(r.id),run=protocol.runs.find((p)=>p.model===selected && p.reward===reward);
    const data=progress.runs.find((p)=>p.model===selected && p.reward===reward);
    const baseline=data?.validation.find((v)=>v.step===0),latest=data?.validation.at(-1);
    const valid=latest && baseline
      ? `Step ${latest.step} · ${latest.mean.toFixed(4)} (${latest.mean-baseline.mean>=0?'+':''}${(latest.mean-baseline.mean).toFixed(4)}) · ${latest.n}/30`
      : '待全量30条验证';
    return row([`${model.name}-${r.name}-3e-4`,r.definition,
      run?`${data?.complete?'完成':status[run.status]||run.status} · ${run.host}${run.project?' / '+run.project:''}${run.job_id?' · '+run.job_id:''}`:'候选，未启动',
      run?`${data?.steps||0}/270`:'—',run?valid:'—','封存 · 未运行']);
  }));
  $('validation').replaceChildren(...progress.runs.filter((r)=>r.model===selected).flatMap((r)=>r.validation.map((v)=>row([
    `${model.name}-${r.reward}-3e-4`,`${v.step} · ${v.n}/30`,
    ...['SP_Richness','Instrumental','Arrangement','Mixing','Musicality'].map((key)=>Number.isFinite(v.cross_means?.[key])?v.cross_means[key].toFixed(4):'n/a'),
    `${v.invalid_n ?? 'n/a'}/30`,
  ]))));
  draw();
}
try {
  const response = await fetch('./data/reward-regime-protocol.json', { cache: 'no-store' });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  protocol = await response.json();
  const progressResponse=await fetch('./data/reward-regime-progress.json',{cache:'no-store'});
  if(!progressResponse.ok)throw Error(`HTTP ${progressResponse.status}`);
  progress=await progressResponse.json();
  if(!protocol.training_authorized || protocol.runs.length!==6)throw Error('Approved matrix mismatch');
  $('updated').textContent=`已核验快照 ${new Date(progress.updated_at).toLocaleString('zh-CN',{timeZone:'Australia/Perth'})} AWST · 单个 LR3e-4 / seed42 · 旧评估和本项目旧训练已暂停；不影响其他人的任务。快照不是远端实时状态。`;
  $('hypothesis').textContent = protocol.hypothesis;
  $('models').replaceChildren(...protocol.models.map((m) => {
    const b = document.createElement('button'); b.type = 'button'; b.textContent = m.name; b.dataset.model = m.id;
    b.addEventListener('click', () => { selected = m.id; render(); }); return b;
  }));
  $('reward-controls').replaceChildren(...protocol.rewards.map((r)=>{
    const b=document.createElement('button');b.type='button';b.textContent=r.name;b.dataset.reward=rewardKey(r.id);
    b.addEventListener('click',()=>{target=b.dataset.reward;render();});return b;
  }));
  $('rewards').replaceChildren(...protocol.rewards.map((r) => {
    const link = document.createElement('a'); link.href = r.source; link.textContent = r.name;
    return row([link, r.definition, r.meaning, `${r.status}。${r.note}`]);
  }));
  $('data').replaceChildren(...['train', 'valid', 'test', 'modalities'].map((key) => row([key, protocol.data[key]])));
  $('design').replaceChildren(...Object.entries(protocol.design).map(([key, value]) => row([key, Array.isArray(value) ? value.join(' / ') : value])));
  new ResizeObserver(draw).observe($('reward-chart'));
  render();
} catch (error) { $('model-note').textContent = '协议加载失败，未显示替代结果。'; console.error(error); }
