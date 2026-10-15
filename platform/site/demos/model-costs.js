const root=document.querySelector('#view-model-costs');
const $=(name)=>root.querySelector(`[data-cost="${name}"]`);
const el=(tag,text,cls)=>{const e=document.createElement(tag);if(text)e.textContent=text;if(cls)e.className=cls;return e;};
const number=(value)=>value.toLocaleString('en-US');
const time=(value)=>{
  if(!Number.isFinite(value))return 'n/a';
  if(value<10)return `0分${value.toFixed(2)}秒`;
  const total=Math.round(value);return `${Math.floor(total/60)}分${String(total%60).padStart(2,'0')}秒`;
};
const rate=(value)=>Number.isFinite(value)?value.toFixed(value<.01?4:3):'n/a';
const cell=(main,sub)=>{const td=el('td');td.append(el('strong',main));if(sub)td.append(el('small',sub,'reference-category'));return td;};
const labels={measured:'实测','linear-estimate':'线性估算','author-rate-estimate':'作者RTF换算',unsupported:'原生不支持','boundary-unverified':'口径未核实 · 撤回外推'};
let data,filter='all';
function render(){
  const rows=data.timing.filter(r=>filter==='all'||(filter==='measured'?r.kind.startsWith('measured'):r.kind==='author-reference'));
  $('timing').replaceChildren(...rows.map(r=>{
    const model=data.models.find(m=>m.id===r.id),tr=el('tr');tr.dataset.costModel=r.id;
    const name=r.name||model.name;
    const kind=r.kind==='measured-rollout-proxy'?'训练worker / rollout实测 · 非纯推理':r.publish_duration_estimates===false?'官方声明 · 完整链路未核实':r.kind==='author-reference'?'文献 / 官方参考 · 非本站实测':r.kind==='unavailable'?'n/a · 当前checkpoint未测':'独立实测';
    tr.append(cell(name,kind),cell(r.hardware,r.backend));
    const direct=r.rows.filter(v=>[60,300].includes(v.target_seconds)&&Number.isFinite(v.rtf));
    tr.append(cell(Number.isFinite(r.median_rtf)?rate(r.median_rtf):direct.length?direct.map(v=>`${v.target_seconds}s: ${rate(v.rtf)}`).join(' / '):'n/a',r.n_label||(r.n?`n=${r.n}${r.median_rtf?' · 中位数':''}`:'')));
    for(const seconds of [60,300]){
      const sample=r.rows.find(v=>v.target_seconds===seconds);
      const type=sample?.kind||'measured';
      tr.append(cell(time(sample?.wall_seconds),sample?[labels[type],type==='measured'&&Number.isFinite(sample.actual_audio_seconds)?`输出${sample.actual_audio_seconds.toFixed(2)}s`:null,sample.caveat].filter(Boolean).join(' · '):'尚无有效计时，未虚构RTF'));
    }
    return tr;
  }));
  root.querySelectorAll('[data-cost-filter]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.costFilter===filter)));
}
try{
  const response=await fetch('./data/model-costs.json?v=20261015-boundaries',{cache:'no-store'});
  if(!response.ok)throw Error(`HTTP ${response.status}`);
  data=await response.json();
  $('timing-audit').replaceChildren(...data.timing_audit.sections.map(s=>{
    const section=el('section');section.append(el('h4',s.title));
    for(const statement of s.statements)section.append(el('p',statement,'section-copy'));
    return section;
  }),...data.timing_audit.sources.map(s=>{
    const p=el('p'),a=el('a',s.label);a.href=s.url;a.target='_blank';a.rel='noopener noreferrer';p.append(a);return p;
  }));
  $('parameters').replaceChildren(...data.models.map(m=>{
    const tr=el('tr');tr.dataset.costModel=m.id;
    const identity=cell(m.name,m.checkpoint);
    const source=el('a','模型来源');source.href=m.source;source.target='_blank';source.rel='noopener noreferrer';identity.append(source);
    tr.append(identity,cell(`${(m.parameters/1e9).toFixed(4)}B`,`${number(m.parameters)} 参数`),
      cell(`${(m.lora_parameters/1e6).toFixed(3)}M · ${m.lora_percent.toFixed(3)}%`,`${number(m.lora_parameters)}参数 / ${m.lora_modules}模块`));
    const scope=cell(m.components,m.plan);tr.append(scope);return tr;
  }));
  $('evidence').replaceChildren(...data.timing.map(r=>{
    const section=el('section'),m=data.models.find(m=>m.id===r.id);
    section.append(el('h4',r.name||m.name),el('p',r.scope,'section-copy'));
    if(r.duration_control)section.append(el('p',`长度控制：${r.duration_control}。本轮是生成吞吐测试，不是完整乐曲质量评测。`,'section-copy'));
    if(Number.isFinite(r.load_seconds))section.append(el('p',`模型初始化 ${time(r.load_seconds)}，与热推理分开；不含集群排队。`,'section-copy'));
    if(r.measured_audio_range)section.append(el('p',`实际片长 ${r.measured_audio_range.map(v=>v.toFixed(2)).join('–')}s；RTF范围 ${r.rtf_range.map(rate).join('–')}。`,'section-copy'));
    if(r.source){const a=el('a','来源 / 计时回执');a.href=r.source.startsWith('http')?r.source:`https://github.com/zhanh-he/RL-paper-reading/blob/main/music-gen/lyrics2song/model-costs/${r.source}`;a.target='_blank';a.rel='noopener noreferrer';section.append(a);}
    for(const item of r.sources||[]){const p=el('p'),a=el('a',item.label);a.href=item.url;a.target='_blank';a.rel='noopener noreferrer';p.append(a);section.append(p);}
    return section;
  }),...data.references.map(r=>{const p=el('p');const a=el('a',r.label);a.href=r.url;a.target='_blank';a.rel='noopener noreferrer';p.append(a);return p;}));
  root.querySelectorAll('[data-cost-filter]').forEach(b=>b.addEventListener('click',()=>{filter=b.dataset.costFilter;render();}));
  $('acceleration').replaceChildren(...data.acceleration.models.map(m=>{
    const tr=el('tr');tr.dataset.accelerationModel=m.id;
    tr.append(cell(m.name,m.engine),cell(m.support));
    for(const seconds of [60,300]){
      const row=m.measured_rows?.find(r=>r.target_seconds===seconds);
      tr.append(cell(time(row?.wall_seconds),row?`实测输出${row.actual_audio_seconds.toFixed(2)}s`:
        m.id==='levo2-medium-reference'&&seconds===300?'超官方270s上限':'尚无可比端到端计时'));
    }
    tr.append(cell(m.measured_rows?.length?m.measured_rows.map(r=>`${r.target_seconds}s: ${r.e2e_speedup.toFixed(2)}×`).join(' / '):'n/a',m.speedup_status));
    return tr;
  }));
  $('projections').replaceChildren(...data.acceleration.projections.map(r=>{
    const tr=el('tr');tr.append(cell(`${r.name} · ${r.target_seconds}s`,r.hardware),
      cell(time(r.baseline_seconds),`AR ${(r.ar_fraction*100).toFixed(1)}%；其余${r.unchanged_seconds.toFixed(1)}s`));
    for(const scenario of r.scenarios)tr.append(cell(time(scenario.estimated_wall_seconds),`条件估算 · 整链${scenario.estimated_e2e_speedup.toFixed(2)}×`));
    tr.append(cell(`${r.ar_only_infinite_speedup_bound.toFixed(2)}×`,'仅AR无限快、其他阶段不变；不是完整Omni的上界'));
    return tr;
  }));
  $('acceleration-evidence').replaceChildren(...data.acceleration.models.map(m=>{
    const section=el('section');section.append(el('h4',m.name),el('p',m.coverage,'section-copy'));
    if(m.measurement_scope)section.append(el('p',m.measurement_scope,'section-copy'));
    const a=el('a','后端源码 / 支持状态来源');a.href=m.source;a.target='_blank';a.rel='noopener noreferrer';section.append(a);
    if(m.receipt){const p=el('p'),link=el('a','本轮完整计时回执');link.href=`https://github.com/zhanh-he/RL-paper-reading/blob/main/music-gen/lyrics2song/model-costs/${m.receipt}`;link.target='_blank';link.rel='noopener noreferrer';p.append(link);section.append(p);}
    return section;
  }));
  render();
}catch(error){const tr=el('tr'),td=el('td',`统计加载失败：${error.message}`);td.colSpan=5;tr.append(td);$('timing').replaceChildren(tr);}
