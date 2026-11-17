import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { checkDeck, csvRows, FIELDS, fresh, restore, toCsv, validateEvent } from '../site/demos/singer-blind/model.mjs';

const path = new URL('../site/demos/singer-blind/',import.meta.url);
const deck = checkDeck(JSON.parse(readFileSync(new URL('deck.json',path),'utf8')));
const first=deck.pairs[0], id=first.first;

test('expanded deck preserves masked publication boundary',()=>{
  assert.deepEqual(Object.keys(deck).sort(),['batch_id','clips','condition_count','family_count','pairs','schema','unique_recordings']);
  assert.deepEqual([deck.unique_recordings,deck.condition_count,deck.family_count,deck.pairs.length,Object.keys(deck.clips).length],[96,24,12,64,102]);
  const mixSha = new Set();
  for (const clip of Object.values(deck.clips)) {
    assert.deepEqual(Object.keys(clip).sort(),['duration_s','lyrics','mix','style','vocal']);
    mixSha.add(clip.mix.sha256);
  }
  assert.equal(mixSha.size,96);
  assert.deepEqual(new Set(deck.pairs.flatMap(p=>[p.first,p.second])),new Set(Object.keys(deck.clips)));
  for (const p of deck.pairs) assert.deepEqual(Object.keys(p).sort(),['first','id','second']);
});

test('blank template contains every issued clip and exact A/B orientation',()=>{
  const lines=readFileSync(new URL('annotations.csv',path),'utf8').trimEnd().split(/\r?\n/);
  assert.equal(lines.shift(),FIELDS.join(','));
  assert.equal(lines.length,102+64);
  assert.deepEqual(lines.filter(s=>s.split(',')[1]==='pair').map(s=>s.split(',').slice(0,4)),
                   deck.pairs.map(p=>[deck.batch_id,'pair',p.first,p.second]));
});

test('real-format clip, pair and event export round-trip without dropped orientation',()=>{
  const draft=fresh(deck);
  draft.selected.pair=first.id;
  draft.clips[id]={identity_judgment:'drift',lyrics_pass:'no',notes:'变化, near "line 2"'};
  draft.events.push(validateEvent(deck,{key:'970a423f-caec-4703-9b40-7a6270f93241',blind_id:id,event_start_s:15.3,event_end_s:19.2,event_label:'identity_drift',notes:''}));
  draft.pairs[first.id]={identity_preference:'second',overall_preference:'tie'};
  const restored=restore(deck,JSON.parse(JSON.stringify(draft)));
  assert.deepEqual(restored,draft);
  const rows=csvRows(deck,restored);
  assert.deepEqual(rows.map(r=>r.label_type),['clip','event','pair']);
  assert.deepEqual([rows[2].blind_id,rows[2].other_blind_id,rows[2].identity_preference],[first.first,first.second,'second']);
  assert.ok(toCsv(rows).includes('"变化, near ""line 2"""'));
});

test('reject wrong deck and unsafe or out-of-bounds events',()=>{
  const draft=fresh(deck);
  assert.throws(()=>restore(deck,{...draft,batch_id:'confirm'}),/其他批次/);
  assert.throws(()=>restore(deck,{...draft,pairs:{P999:{identity_preference:'first'}}}),/未知/);
  const base={key:'970a423f-caec-4703-9b40-7a6270f93241',blind_id:id,event_start_s:2,event_end_s:3,event_label:'missing_vocal'};
  for (const e of [{event_start_s:-1},{event_end_s:1},{event_end_s:1000},{blind_id:'L00000000'},{event_label:'pretend_truth'}]) {
    assert.throws(()=>validateEvent(deck,{...base,...e}));
  }
  assert.deepEqual(csvRows(deck,draft),[]);
  assert.ok(toCsv([{notes:'=123'}]).includes('"\'=123"'));
});
