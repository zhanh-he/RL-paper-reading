import assert from 'node:assert/strict';
import fs from 'node:fs';
const root=new URL('../../music-gen/reward-regime-study/runs/2026-10-20-songbench-review/',import.meta.url);
const read=name=>JSON.parse(fs.readFileSync(new URL(name,root),'utf8'));
const audit=read('verified-recovery.json'),listening=read('listening.json');
assert.equal(audit.new_training_started,false);
assert.equal(audit.runs.length,6);
let seed=20261020;
const random=()=>{seed^=seed<<13;seed^=seed>>>17;seed^=seed<<5;return (seed>>>0)/4294967296;};
const runs=audit.runs.map(r=>{
  assert.equal(r.status,'verified-historical-comparator');
  assert.equal(r.updates,270);
  assert.equal(r.binding.bound_audio_n,660);
  assert.deepEqual(r.validation.map(v=>[v.step,v.n]),[[0,30],[50,30],[100,30],[270,30]]);
  const before=r.validation[0],after=r.validation.at(-1);
  const deltas=after.rows.map((row,i)=>row.mean7-before.rows[i].mean7);
  const boot=Array.from({length:10000},()=>Array.from({length:30},()=>deltas[Math.floor(random()*30)]).reduce((a,b)=>a+b,0)/30).sort((a,b)=>a-b);
  return {id:r.run,model:r.model,reward:r.experiment.reward,experiment:r.experiment,
    updates:r.updates,trace:r.trace,kl_unit:r.kl_unit,
    validation:r.validation.map(v=>({step:v.step,n:v.n,target_mean:v.target_mean,means:v.means,invalid_n:v.invalid_n})),
    mean7_delta:after.means.SongBench_mean7-before.means.SongBench_mean7,
    mean7_delta_ci95:[boot[250],boot[9749]],binding:r.binding,
    experiment_sha256:r.experiment_sha256,steps_sha256:r.steps_sha256,
    validation_receipt_sha256:r.validation.map(v=>({step:v.step,sha256:v.receipt_sha256}))};
});
assert.equal(listening.rows.length,20);
for(const row of listening.rows){
  assert.match(row.url,/\.mp3$/);
  assert.ok(row.source_audio_sha256&&row.public_mp3_sha256);
}
const value={updated_at:audit.checked_at,new_training_started:false,
  objective:'R = (Vocal + Instrumental + Melody + Structure + Arrangement + Mixing + Musicality) / 7; only SongBench, no extra penalty; KL separate.',
  objective_status:'User-confirmed definition; new execution matrix awaiting approval. No historical Mean7-trained arm.',
  scope:'Historical all-30 validation, not test or human MOS. Mean7 is a retrospective cross-score, NOT the historical optimized objective. Paired condition-bootstrap 95% intervals, one training seed, uncorrected multiple comparisons.',
  failure:'All six new Muse/HeartMuLa backbone jobs failed before training: scorer import ModuleNotFoundError: generate_objective. Scheduler IDs no longer present; not still queued.',
  priorities:[
    ['P1 · first','DiffRhythm2–SongBenchMean7–3e-4','Miles主干链已验收；改接唯一Mean7、固定60s、G2 /beta0.05 /rank8 /270步；先做新reward接口验收。'],
    ['P1 · paired','YuE2–SongBenchMean7–3e-4','强起点参照；TRL原生codec策略及60s训练尚待验收，不能把旧PyTorch短样本结果改名填入。'],
    ['P2','Muse–SongBenchMean7–3e-4','先修评分服务依赖打包，再验证主干/codec路径；不恢复旧子项任务。'],
    ['P3','HeartMuLa → SongBloom / ACE','首轮不扩散模型和LR网格；这些模型不阻塞P1结果。']
  ],
  runs,listening:listening.rows,
  audit_url:'https://github.com/zhanh-he/RL-paper-reading/blob/main/music-gen/reward-regime-study/runs/2026-10-20-songbench-review/verified-recovery.json'};
const destination=new URL('../site/demos/data/songbench-review.json',import.meta.url);
const text=JSON.stringify(value,null,2)+'\n';
if(process.argv.includes('--check'))assert.equal(fs.readFileSync(destination,'utf8'),text);
else fs.writeFileSync(destination,text);
console.log('SongBench review: six verified historical arms, 3960 SHA-bound audio records, 20 MP3 replays; new Mean7 training NOT started.');
