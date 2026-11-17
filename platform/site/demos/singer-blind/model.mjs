export const FIELDS = 'batch_id,label_type,blind_id,other_blind_id,event_start_s,event_end_s,event_label,identity_judgment,identity_preference,overall_preference,lyrics_pass,accompaniment_pass,expression_pass,confidence,notes'.split(',');
export const CLIP = {identity_judgment:['stable','drift','unsure'],lyrics_pass:['yes','no','unsure'],accompaniment_pass:['yes','no','unsure'],expression_pass:['yes','no','unsure']};
export const PAIR = {identity_preference:['first','second','tie','unsure'],overall_preference:['first','second','tie','unsure']};
export const EVENTS = ['identity_drift','normal_expression','missing_vocal','separation_artifact','extra_vocal','wrong_role','unsure'];

const idFormat = /^L[0-9A-F]{8}$/;
const object = value => value && typeof value === 'object' && !Array.isArray(value);
const own = (value, key) => Object.hasOwn(value, key);

export function checkDeck(deck) {
  if (!object(deck) || deck.schema !== 1 || !/^[a-z0-9-]+$/.test(deck.batch_id) || !object(deck.clips) || !Array.isArray(deck.pairs)) throw Error('批次数据格式错误');
  const pairIds = new Set();
  for (const [id, clip] of Object.entries(deck.clips)) {
    if (!idFormat.test(id) || !(Number.isFinite(clip.duration_s) && clip.duration_s > 0) || typeof clip.style !== 'string' || typeof clip.lyrics !== 'string') throw Error('无效的音频记录');
    for (const [role, suffix] of [['mix',''],['vocal','-v']]) {
      const asset = clip[role];
      if (asset?.file !== `${id}${suffix}.mp3` || !/^[a-f0-9]{64}$/.test(asset.sha256) || !(asset.bytes > 0)) throw Error('无效的 MP3 绑定');
    }
  }
  for (const p of deck.pairs) {
    if (!/^P\d{2,}$/.test(p.id) || pairIds.has(p.id) || p.first === p.second || !own(deck.clips,p.first) || !own(deck.clips,p.second)) throw Error('无效的试听配对');
    const a = deck.clips[p.first], b = deck.clips[p.second];
    if (a.style !== b.style || a.lyrics !== b.lyrics) throw Error('试听对应不同输入');
    pairIds.add(p.id);
  }
  if (!pairIds.size) throw Error('没有可用的试听对');
  return deck;
}

export function fresh(deck) {
  return {schema:1,batch_id:deck.batch_id,mode:'pair',selected:{pair:deck.pairs[0].id,clip:Object.keys(deck.clips)[0]},clips:{},pairs:{},events:[]};
}

const cleanNotes = notes => {
  if (typeof notes !== 'string' || notes.length > 1000) throw Error('备注最多 1000 字');
  return notes;
};

export function validateEvent(deck, e) {
  if (!object(e) || typeof e.key !== 'string' || !/^[a-f0-9-]{20,80}$/.test(e.key) || !own(deck.clips,e.blind_id) || !EVENTS.includes(e.event_label)) throw Error('无法识别事件或曲目');
  if (!Number.isFinite(e.event_start_s) || !Number.isFinite(e.event_end_s) || e.event_start_s < 0 || e.event_start_s >= e.event_end_s || e.event_end_s > deck.clips[e.blind_id].duration_s + 1e-8) throw Error('需满足 0 ≤ 起点 < 终点 ≤ 音频长度');
  return {key:e.key,blind_id:e.blind_id,event_start_s:e.event_start_s,event_end_s:e.event_end_s,event_label:e.event_label,notes:cleanNotes(e.notes ?? '')};
}

function importScores(source, ids, spec) {
  if (!object(source)) throw Error('标注记录格式错误');
  const out = {};
  for (const [id, value] of Object.entries(source)) {
    if (!ids.has(id) || !object(value)) throw Error('标注包含未知的曲目或配对');
    const row = {};
    for (const [field, pick] of Object.entries(value)) {
      if (field === 'notes') row.notes = cleanNotes(pick);
      else if (own(spec,field) && (pick === '' || spec[field].includes(pick))) row[field] = pick;
      else throw Error('标注包含无效选项');
    }
    out[id] = row;
  }
  return out;
}

export function restore(deck, state) {
  if (!object(state) || state.schema !== 1 || state.batch_id !== deck.batch_id) throw Error('备份属于其他批次或版本');
  const out = fresh(deck), ids = new Set(Object.keys(deck.clips)), pids = new Set(deck.pairs.map(p => p.id));
  out.mode = state.mode === 'clip' ? 'clip' : 'pair';
  if (ids.has(state.selected?.clip)) out.selected.clip = state.selected.clip;
  if (pids.has(state.selected?.pair)) out.selected.pair = state.selected.pair;
  out.clips = importScores(state.clips,ids,CLIP);
  out.pairs = importScores(state.pairs,pids,PAIR);
  if (!Array.isArray(state.events) || state.events.length > 5000) throw Error('事件表格式错误');
  const keys = new Set();
  out.events = state.events.map(event => {
    const e = validateEvent(deck,event);
    if (keys.has(e.key)) throw Error('重复事件 ID');
    keys.add(e.key);
    return e;
  });
  return out;
}

export function csvRows(deck, state) {
  const result = [], base = {batch_id:deck.batch_id};
  const filled = r => r && Object.values(r).some(value => value !== '');
  for (const [id,row] of Object.entries(state.clips)) {
    if (filled(row)) result.push({...base,label_type:'clip',blind_id:id,...row});
  }
  for (const e of state.events) {
    const {key, ...row} = e;
    result.push({...base,label_type:'event',...row});
  }
  for (const {id,first,second} of deck.pairs) {
    if (filled(state.pairs[id])) result.push({...base,label_type:'pair',blind_id:first,other_blind_id:second,...state.pairs[id]});
  }
  return result;
}

export function toCsv(rows) {
  const safe = (cell, field) => {
    const str = String(cell ?? '');
    const guarded = field === 'notes' && /^[\s]*[=+@-]/.test(str) ? `'${str}` : str;
    return `"${guarded.replaceAll('"','""')}"`;
  };
  return [FIELDS.join(','), ...rows.map(r => FIELDS.map(field => safe(r[field],field)).join(','))].join('\r\n')+'\r\n';
}
