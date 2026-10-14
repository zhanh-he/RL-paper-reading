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
let protocol, progress, recovery, selected = 'yue2', target = 'richness', replayMode='long', example=0, replayStep=270, badId=0;
const rewardKey = (id) => ({ songbench: 'instrument', 'songbench-arrangement': 'arrangement' })[id] || id;
const status = { running: '运行中', 'local-sequential-queue': '本机连续队列', 'pbs-queued': 'PBS 排队（未起跑）' };
const rewardName=(key)=>({richness:'Richness',instrument:'SongBench-Instrument',arrangement:'SongBench-Arrangement'})[key]||key;
const runName=(key)=>`${selected==='yue2'?'YuE2':'Muse'}-${rewardName(key)}-3e-4`;
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

function buttons(container,items,current,choose) {
  container.replaceChildren(...items.map(([value,label])=>{
    const b=document.createElement('button');b.type='button';b.textContent=label;b.setAttribute('aria-pressed',String(value===current));
    b.addEventListener('click',()=>choose(value));return b;
  }));
}

function panel(asset,label) {
  const article=document.createElement('article');article.className='compare-panel';
  const h=document.createElement('h3');h.textContent=label;
  const wave=document.createElement('img');wave.src=asset.wave;wave.alt=`${label} waveform, fixed -1 to +1`;wave.loading='lazy';wave.className='regime-wave';
  const spectrum=document.createElement('img');spectrum.src=asset.spectrum;spectrum.alt=`${label} mel spectrum, 0-24kHz, -80 to 0dB, black floor`;spectrum.loading='lazy';spectrum.className='regime-mel';
  const audio=document.createElement('audio');audio.controls=true;audio.preload='metadata';audio.src=asset.audio;
  const scale=document.createElement('small');scale.textContent='Wave −1…+1 · Mel 0–24kHz / −80…0dB（固定）';
  audio.dataset.sourceSha=asset.source_audio_sha256;
  audio.addEventListener('play',()=>document.querySelectorAll('audio').forEach((a)=>{if(a!==audio)a.pause();}));
  const note=document.createElement('small');note.textContent=`${asset.seconds.toFixed(2)}s · MP3 192kbps / 48kHz · 无响度归一化 · 原始 SHA ${asset.source_audio_sha256.slice(0,12)} · MP3 SHA ${asset.sha256.slice(0,12)}`;
  note.title=`Original ${asset.source_audio_sha256}\nMP3 ${asset.sha256}`;
  article.append(h,wave,spectrum,scale,audio,note);return article;
}

function renderMedia() {
  if(!recovery)return;
  for(const b of $('replay-modes').querySelectorAll('button'))b.setAttribute('aria-pressed',String(b.dataset.mode===replayMode));
  $('media').querySelectorAll('audio').forEach((a)=>{a.pause();a.removeAttribute('src');a.load();});
  $('media').replaceChildren();$('pair-metrics').replaceChildren();$('example-controls').replaceChildren();$('step-controls').replaceChildren();
  if(selected!=='yue2'){$('replay-note').textContent='Muse 尚无已回收的本组正式训练音频；不使用其他实验的音频替代。';return;}
  let a,b,step,index;
  if(replayMode==='long') {
    const available=recovery.long_replays.filter((r)=>r.reward===target);
    if(!available.length){$('replay-note').textContent=`${runName(target)} 尚无已回收长片段。已完成的曲线/验证分数仍显示在上方。`;return;}
    const indices=[...new Set(available.map((r)=>r.index))];if(!indices.includes(example))example=indices[0];
    const steps=[0,...new Set(available.filter((r)=>r.index===example).map((r)=>r.step))].sort((x,y)=>x-y);
    if(!steps.includes(replayStep))replayStep=steps.at(-1);
    buttons($('example-controls'),indices.map((i)=>[i,`CMI valid #${i+1}`]),example,(i)=>{example=i;renderMedia();});
    buttons($('step-controls'),steps.map((s)=>[s,`${s} steps`]),replayStep,(s)=>{replayStep=s;renderMedia();});
    a=recovery.baseline.find((r)=>r.index===example).asset;
    b=replayStep===0?a:available.find((r)=>r.index===example&&r.step===replayStep).asset;step=replayStep;index=example;
    $('replay-note').textContent=`${runName(target)} · 同一 Text+Lyrics、seed ${5101+index}；A 始终为冻结0步。原生长片段（750–1500 semantic tokens），末尾可能因预算截断，无循环/补齐；不是短预算30条验证的均值。参考音频未输入，GT为n/a。图像来自MP3，以下分数来自其绑定的原始lossless音频。`;
  } else {
    const cases=recovery.badcases.filter((r)=>r.reward===target);
    if(!cases.length){$('replay-note').textContent='尚无符合本次冲突筛选规则的已回收候选，不编造 badcase；可查看全量验证与选择规则。';return;}
    if(badId>=cases.length)badId=0;
    buttons($('example-controls'),cases.map((r,i)=>[i,`Valid #${r.index+1} · step ${r.step}`]),badId,(i)=>{badId=i;renderMedia();});
    const item=cases[badId];a=item.baseline;b=item.candidate;step=item.step;index=item.index;
    const intermediate=recovery.audits.find((r)=>r.reward===target)?.complete_arm?'':'该臂仍训练中，这是中间checkpoint，不是最终结果。';
    $('replay-note').textContent=`${runName(target)} · 验证 #${index+1} / step ${step}。${intermediate}这是确实被评分的短验证音频，保留原始长度，不伪装成30–60秒长回放。刻意挑选“目标上升、两个交叉维度下降”的案例，尚未由人确认听感退化，不代表总体比例或已证实reward hacking。`;
  }
  $('media').append(panel(a,'A · Baseline · 0 steps'),panel(b,`B · ${runName(target)} · ${step} steps`));
  const value=(m,key)=>key==='objective'?(target==='richness'?m.SP_Richness:m[target==='instrument'?'Instrumental':'Arrangement']/10)-(m.penalty??0):m[key];
  $('pair-metrics').replaceChildren(...[['objective','Target − shared penalty'],['SP_Richness','SP Richness · 0–1'],['Instrumental','SongBench Instrumental · 0–10'],['Arrangement','SongBench Arrangement · 0–10'],['Musicality','SongBench Musicality · 0–10'],['Mixing','SongBench Mixing · 0–10'],['clipping_fraction','原始音频近满幅样本比例']].map(([key,label])=>{
    const x=value(a.metrics,key),y=value(b.metrics,key),delta=y-x;
    return row([label,x.toFixed(4),`${y.toFixed(4)} (${delta>=0?'+':''}${delta.toFixed(4)})`]);
  }));
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
    return row([['yue2','muse'].includes(selected)?runName(reward):`${model.name}-${r.name}`,r.definition,
      run?`${data?.complete?'完成':status[run.status]||run.status} · ${run.host}${run.project?' / '+run.project:''}${run.job_id?' · '+run.job_id:''}`:'候选，未启动',
      run?`${data?.steps||0}/270`:'—',run?valid:'—','封存 · 未运行']);
  }));
  $('validation').replaceChildren(...progress.runs.filter((r)=>r.model===selected).flatMap((r)=>r.validation.map((v)=>row([
    `${model.name}-${r.reward}-3e-4`,`${v.step} · ${v.n}/30`,
    ...['SP_Richness','Instrumental','Arrangement','Mixing','Musicality'].map((key)=>Number.isFinite(v.cross_means?.[key])?v.cross_means[key].toFixed(4):'n/a'),
    `${v.invalid_n ?? 'n/a'}/30`,
  ]))));
  draw();
  renderMedia();
}
try {
  const response = await fetch('./data/reward-regime-protocol.json', { cache: 'no-store' });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  protocol = await response.json();
  const progressResponse=await fetch('./data/reward-regime-progress.json',{cache:'no-store'});
  if(!progressResponse.ok)throw Error(`HTTP ${progressResponse.status}`);
  progress=await progressResponse.json();
  const recoveryResponse=await fetch('./data/reward-regime-recovery.json',{cache:'no-store'});
  if(!recoveryResponse.ok)throw Error(`Media HTTP ${recoveryResponse.status}`);
  recovery=await recoveryResponse.json();
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
  $('replay-modes').querySelectorAll('button').forEach((b)=>b.addEventListener('click',()=>{replayMode=b.dataset.mode;renderMedia();}));
  const interval=(v)=>`${v.mean_delta>=0?'+':''}${v.mean_delta.toFixed(4)} [${v.ci95.map((x)=>x.toFixed(4)).join(', ')}]`;
  $('case-audit').replaceChildren(...recovery.audits.map((r)=>row([`${rewardName(r.reward)} · ${r.step}${r.complete_arm?' · 完整270步':' · 训练中间checkpoint'}`,`${r.candidate_n}/${r.n}`,
    ...['objective','Instrumental','Arrangement'].map((key)=>interval(r.statistics[key])),r.selection])));
  new ResizeObserver(draw).observe($('reward-chart'));
  render();
} catch (error) { $('model-note').textContent = '协议加载失败，未显示替代结果。'; console.error(error); }
