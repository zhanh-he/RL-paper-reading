import {checkDeck, fresh, restore, validateEvent, csvRows, toCsv, CLIP, PAIR, EVENTS} from './model.mjs';

const $ = id => document.getElementById(id);
const q = (node, selector) => node.querySelector(selector);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const text = {stable:'同一歌手',drift:'有换人／漂移',unsure:'不确定',yes:'通过',no:'不通过',
  first:'A',second:'B',tie:'无差别', identity_drift:'真实换人／身份漂移',normal_expression:'正常换声区／表现力变化',
  missing_vocal:'漏唱／主唱缺失',separation_artifact:'疑似分离伪影',extra_vocal:'额外人声',wrong_role:'歌手角色错误'};
const clipFields = {identity_judgment:'歌手身份',lyrics_pass:'歌词覆盖',accompaniment_pass:'伴奏',expression_pass:'表现力'};
const pairFields = {identity_preference:'哪版歌手更稳定？',overall_preference:'整体更喜欢哪版？'};
let deck,state,key;
const assetRoot = 'https://github.com/zhanh-he/RL-paper-reading/releases/download/';
const url = asset => `${assetRoot}${deck.batch_id}/${asset.file}`;

function progress() {
  const pairN = Object.values(state.pairs).filter(a => a.identity_preference).length;
  const clipN = Object.values(state.clips).filter(a => a.identity_judgment).length;
  $('pair-progress').textContent = `身份配对 ${pairN}/${deck.pairs.length}`;
  $('clip-progress').textContent = `单曲判断 ${clipN}/${Object.keys(deck.clips).length}`;
  $('event-progress').textContent = `事件 ${state.events.length}`;
  $('export-csv').disabled = $('copy-csv').disabled = !csvRows(deck,state).length;
}

function save() {
  try {
    localStorage.setItem(key,JSON.stringify(state));
    $('storage-status').textContent = '本机已保存';
  } catch {
    $('storage-status').textContent = '浏览器存储不可用，请导出备份';
  }
  progress();
}

function options(values,pick) {
  return '<option value="">未标</option>'+values.map(v => `<option value="${v}" ${v===pick?'selected':''}>${text[v]}</option>`).join('');
}

function eventList(id) {
  return state.events.filter(e => e.blind_id===id).sort((a,b)=>a.event_start_s-b.event_start_s).map(e =>
    `<div class="event-item"><span>${e.event_start_s.toFixed(2)}–${e.event_end_s.toFixed(2)} s · ${text[e.event_label]}${e.notes?' · '+esc(e.notes):''}</span><div><button type="button" data-edit="${e.key}">编辑</button> <button type="button" data-remove="${e.key}">删除</button></div></div>`).join('');
}

function take(id, side, single) {
  const c = deck.clips[id], a = state.clips[id] ?? {};
  const labels = Object.entries(CLIP).map(([name,values]) =>
    `<label>${clipFields[name]}<select data-clip-field="${name}">${options(values,a[name])}</select></label>`).join('');
  return `<article class="take take-${side}" data-clip="${id}">
    <div class="take-title"><h3>${single?'曲目':side.toUpperCase()} <span class="id">${id}</span></h3><span class="duration">${c.duration_s.toFixed(1)} s</span></div>
    <label class="mix-label">完整混音<audio controls preload="none" data-mix aria-label="${id} 完整混音" src="${url(c.mix)}"></audio></label>
    <a class="audio-link" href="${url(c.mix)}" target="_blank" rel="noopener">打开混音 MP3 ↗</a>
    <details class="stem"><summary>分离人声</summary><audio controls preload="none" aria-label="${id} 分离人声" src="${url(c.vocal)}"></audio></details>
    <details class="review" ${single?'open':''}><summary>单曲与事件标注 <span class="event-tally">(${state.events.filter(e=>e.blind_id===id).length})</span></summary>
      <div class="labels">${labels}</div>
      <label class="note-label">曲目备注<textarea maxlength="1000" data-clip-field="notes" rows="1">${esc(a.notes)}</textarea></label>
      <form class="events" data-event-form>
        <h3>时段事件</h3>
        <div class="event-grid">
          <label>起点（秒）<input name="start" type="number" required min="0" max="${c.duration_s}" step="any"><button type="button" data-now="start">用混音位置</button></label>
          <label>终点（秒）<input name="end" type="number" required min="0" max="${c.duration_s}" step="any"><button type="button" data-now="end">用混音位置</button></label>
          <label class="kind">事件类型<select required name="kind"><option value="">选择事件</option>${EVENTS.map(e=>`<option value="${e}">${text[e]}</option>`).join('')}</select></label>
        </div>
        <label class="note-label">事件备注<textarea name="notes" maxlength="1000" rows="1"></textarea></label>
        <div class="actions"><button class="event-submit" type="submit">添加事件</button><button class="event-cancel" type="button" hidden>取消编辑</button></div>
        <p class="event-error" role="alert"></p>
        <div class="event-list">${eventList(id)}</div>
      </form>
    </details></article>`;
}

function pairForm(id) {
  const current = state.pairs[id] ?? {};
  const radios = Object.entries(PAIR).map(([field,values])=>`<fieldset><legend>${pairFields[field]}</legend>
    <div class="choices">${values.map(v=>`<label><input type="radio" name="${field}" data-pair-field="${field}" value="${v}" ${current[field]===v?'checked':''}><span>${text[v]}</span></label>`).join('')}
    <button type="button" data-clear-pair="${field}">清除</button></div></fieldset>`).join('');
  return `<section class="pair-form" aria-label="配对偏好"><div class="pair-questions">${radios}</div>
    <label class="note-label">配对备注<textarea maxlength="1000" data-pair-field="notes" rows="1">${esc(current.notes)}</textarea></label></section>`;
}

const currentItems = () => state.mode === 'pair' ? deck.pairs.map(p=>p.id) : Object.keys(deck.clips);

function render() {
  for (const b of document.querySelectorAll('[data-mode]')) b.setAttribute('aria-pressed',String(b.dataset.mode===state.mode));
  const items=currentItems(), id=state.selected[state.mode], i=items.indexOf(id);
  const jump=$('jump'); jump.replaceChildren(...items.map(id=>new Option(id,id)));
  jump.value=id;
  $('previous').disabled=i<1;
  $('next').disabled=i>=items.length-1;
  const single=state.mode==='clip';
  const pair=single?null:deck.pairs.find(p=>p.id===id), source=deck.clips[pair?.first ?? id];
  $('stage').innerHTML=`<details class="input-context"><summary>${single?id:pair.id} · 输入与预期歌词</summary><p>${esc(source.style)}</p><pre>${esc(source.lyrics)}</pre></details>
    <div class="takes ${single?'single':''}">${take(pair?.first??id,'a',single)}${single?'':take(pair.second,'b',false)}</div>${single?'':pairForm(pair.id)}`;
  progress();
}

function navigate(id) {
  if (!currentItems().includes(id)) return;
  state.selected[state.mode]=id;
  render(); save();
}

function resetEvent(form) {
  form.reset(); delete form.dataset.editing;
  q(form,'.event-submit').textContent='添加事件'; q(form,'.event-cancel').hidden=true;
  q(form,'.event-error').textContent='';
}

function updateEvents(form, id) {
  q(form,'.event-list').innerHTML=eventList(id);
  q(form.closest('[data-clip]'),'.event-tally').textContent=`(${state.events.filter(e=>e.blind_id===id).length})`;
  resetEvent(form); save();
}

function saveField(target) {
  if (target.dataset.clipField) {
    const id=target.closest('[data-clip]').dataset.clip;
    (state.clips[id] ??= {})[target.dataset.clipField]=target.value;
  } else if (target.dataset.pairField) {
    (state.pairs[state.selected.pair] ??= {})[target.dataset.pairField]=target.value;
  } else return;
  save();
}

function download(content, type, name) {
  const href=URL.createObjectURL(new Blob([content],{type}));
  const a=document.createElement('a'); a.href=href; a.download=name;
  document.body.append(a); a.click(); a.remove();
  setTimeout(()=>URL.revokeObjectURL(href),60000);
}

function wire() {
  document.querySelectorAll('[data-mode]').forEach(b=>b.addEventListener('click',()=>{
    if (state.mode!==b.dataset.mode) {state.mode=b.dataset.mode; render(); save();}
  }));
  $('jump').addEventListener('change',e=>navigate(e.target.value));
  for (const [id, delta] of [['previous',-1],['next',1]]) $(id).addEventListener('click',()=>{
    const items=currentItems(); navigate(items[items.indexOf(state.selected[state.mode])+delta]);
  });
  $('unrated').addEventListener('click',()=>{
    const items=currentItems(), i=items.indexOf(state.selected[state.mode]);
    const next=[...items.slice(i+1),...items.slice(0,i+1)].find(id=>state.mode==='pair'?!state.pairs[id]?.identity_preference:!state.clips[id]?.identity_judgment);
    if (next) navigate(next);
    else $('storage-status').textContent='当前视图已标完';
  });
  $('stage').addEventListener('change',e=>saveField(e.target));
  $('stage').addEventListener('input',e=>{if (e.target.matches('textarea[data-clip-field],textarea[data-pair-field]')) saveField(e.target);});
  $('stage').addEventListener('play',e=>{
    for (const a of $('stage').querySelectorAll('audio')) if (a!==e.target) a.pause();
  },true);
  $('stage').addEventListener('error',e=>{
    if (e.target.matches('audio')) $('storage-status').textContent='音频载入失败，可用 MP3 链接重试';
  },true);
  $('stage').addEventListener('click',e=>{
    const b=e.target.closest('button'); if (!b) return;
    if (b.dataset.clearPair) {
      const field=b.dataset.clearPair;
      if (state.pairs[state.selected.pair]) delete state.pairs[state.selected.pair][field];
      for (const r of b.closest('fieldset').querySelectorAll('input')) r.checked=false;
      save(); return;
    }
    const form=b.closest('[data-event-form]'), take=b.closest('[data-clip]');
    if (!form || !take) return;
    const id=take.dataset.clip;
    if (b.dataset.now) {
      const t=Math.min(q(take,'[data-mix]').currentTime,deck.clips[id].duration_s);
      form.elements.namedItem(b.dataset.now).value=(Math.floor(t*100)/100).toFixed(2);
    } else if (b.dataset.remove) {
      state.events=state.events.filter(e=>e.key!==b.dataset.remove); updateEvents(form,id);
    } else if (b.dataset.edit) {
      const e=state.events.find(e=>e.key===b.dataset.edit); if (!e) return;
      form.dataset.editing=e.key;
      form.elements.start.value=e.event_start_s; form.elements.end.value=e.event_end_s;
      form.elements.kind.value=e.event_label; form.elements.notes.value=e.notes;
      q(form,'.event-submit').textContent='保存时段'; q(form,'.event-cancel').hidden=false;
      form.elements.start.focus();
    } else if (b.matches('.event-cancel')) resetEvent(form);
  });
  $('stage').addEventListener('submit',e=>{
    e.preventDefault(); const form=e.target;
    const id=form.closest('[data-clip]').dataset.clip;
    try {
      const ev=validateEvent(deck,{key:form.dataset.editing ?? crypto.randomUUID(),blind_id:id,
        event_start_s:form.elements.start.valueAsNumber,event_end_s:form.elements.end.valueAsNumber,
        event_label:form.elements.kind.value,notes:form.elements.notes.value});
      const i=state.events.findIndex(e=>e.key===ev.key);
      if (i>=0) state.events[i]=ev; else state.events.push(ev);
      updateEvents(form,id);
    } catch (error) {q(form,'.event-error').textContent=error.message;}
  });
  const timestamp=()=>new Date().toISOString().replace(/[:.]/g,'-');
  const csv=()=>toCsv(csvRows(deck,state));
  $('export-csv').addEventListener('click',()=>{
    download(csv(),'text/csv;charset=utf-8',`annotations-${deck.batch_id}-${timestamp()}.csv`);
    $('return-status').textContent='CSV 已导出，请把下载的文件发回聊天。';
  });
  $('export-json').addEventListener('click',()=>{
    download(JSON.stringify(state,null,2)+'\n','application/json',`draft-${deck.batch_id}-${timestamp()}.json`);
    $('return-status').textContent='备份已导出。';
  });
  $('copy-csv').addEventListener('click',async()=>{
    try {
      await navigator.clipboard.writeText(csv());
      $('return-status').textContent='已复制 CSV，可直接粘贴回聊天。';
    } catch {
      $('return-status').replaceChildren();
      const area=document.createElement('textarea'); area.value=csv(); area.readOnly=true;
      area.className='export-text'; area.setAttribute('aria-label','标注 CSV');
      $('return-status').append(area); area.select();
    }
  });
  $('import-json').addEventListener('change',async e=>{
    const file=e.target.files[0]; if (!file) return;
    try {
      if (file.size>8_000_000) throw Error('备份文件过大');
      const next=restore(deck,JSON.parse(await file.text()));
      if (csvRows(deck,state).length && !window.confirm('用备份替换本机草稿？建议先导出当前草稿。')) return;
      state=next; render(); save(); $('return-status').textContent='已恢复备份。';
    } catch (error) {$('return-status').textContent=`没有导入：${error.message}`;}
    finally {e.target.value='';}
  });
}

try {
  const response=await fetch('./deck.json');
  if (!response.ok) throw Error('未找到试听目录');
  deck=checkDeck(await response.json());
  key=`music-rl/blind/${deck.batch_id}`; state=fresh(deck);
  try {
    const stored=localStorage.getItem(key);
    if (stored) {state=restore(deck,JSON.parse(stored));$('storage-status').textContent='已恢复本机草稿';}
  } catch {$('storage-status').textContent='无法读取本机草稿，可导入备份';}
  $('scope').textContent=`${deck.unique_recordings} 条开发音频 · ${deck.pairs.length} 对试听 · ${deck.family_count} 个曲目家族`;
  for (const id of ['jump','export-json','import-json']) $(id).disabled=false;
  wire(); render();
} catch (error) {
  $('stage').textContent=`暂时无法打开盲听：${error.message}`;
  $('stage').classList.add('error');
}
