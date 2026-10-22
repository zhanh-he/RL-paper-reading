import assert from 'node:assert/strict';
import fs from 'node:fs';
const study=new URL('../../music-gen/reward-regime-study/',import.meta.url);
const read=name=>JSON.parse(fs.readFileSync(new URL(name,study),'utf8'));
const weekly=read('weekly-2026-10-23/summary.json'),listening=read('runs/2026-10-20-songbench-review/listening.json');
const launch=read('runs/2026-10-20-songbench-priority/launch.json');
assert.equal(weekly.runs.length,8);
const runs=weekly.runs.map(r=>{
  assert.equal(r.updates,270);assert.equal(r.trace.length,270);assert.ok(r.validation.every(v=>v.n===30));
  return {id:r.id,model:r.model,reward:r.reward,experiment:r.experiment,updates:r.updates,trace:r.trace,kl_unit:r.kl_unit,
    validation:r.validation.map(v=>({step:v.step,n:v.n,target_mean:v.target_mean,means:v.means,invalid_n:v.invalid_n})),
    mean7_delta:r.mean7.delta,mean7_delta_ci95:r.mean7.ci95,binding:r.binding,
    experiment_sha256:r.experiment_sha256,steps_sha256:r.steps_sha256,
    validation_receipt_sha256:r.validation.map(v=>({step:v.step,sha256:v.receipt_sha256}))};
});
assert.equal(runs.reduce((n,r)=>n+r.binding.bound_audio_n,0),5220);assert.equal(listening.rows.length,20);
const value={updated_at:weekly.updated_at,new_training_started:true,mean7_training_started:true,
  objective:'R = (Vocal + Instrumental + Melody + Structure + Arrangement + Mixing + Musicality) / 7; only SongBench, no extra penalty; KL separate.',
  objective_status:'DiffRhythm2 Mean7 and Muse-SP completed270/all30; Muse learned-head arms failed before training; YuE2 TRL Mean7 failed acceptance.',
  execution:`手动回收 ${weekly.updated_at}：Muse-SP、Diff Mean7均已核验270步和全30验证。两组Muse learned-head启动失败；YuE2 TRL验收失败，不再标为排队或运行。详见本周汇报。`,
  base_identity:launch.base,
  launch_url:'https://github.com/zhanh-he/RL-paper-reading/tree/main/music-gen/reward-regime-study/weekly-2026-10-23',
  scope:weekly.method.scope,
  failure:'上一轮generate_objective故障修复后，本轮Muse两组仍因libpython3.11.so.1.0缺失在scorer启动前失败；YuE2 TRL因gradient-checkpointing模型接口不兼容在首次更新前失败。这三组没有正式结果，不再展示为排队。Muse-SP和Diff Mean7完整结果已独立回收；失败不等于模型不能RL。',
  priorities:[
    ['P0 · 完成并核验','Muse–SP Richness–3e-4','270步；0/270各30/30；原始非GRPO起点，beta0.05，10.24s codec。目标增益不等于听感增益。'],
    ['P0 · 启动失败','Muse–SongBench-Instrument / Arrangement–3e-4','181718185 /181718187：缺少libpython3.11.so.1.0；没有optimizer更新。不再标排队。'],
    ['P1 · 完成并核验','DiffRhythm2–SongBenchMean7–3e-4','270步；0/50/100/270各30/30；60s原生预算，Miles主干LoRA。Mean7 5.116→5.055，未确立改善。'],
    ['P1 · 验收失败','YuE2–SongBenchMean7–3e-4','181718819：TRL初始化时gradient checkpointing接口不兼容，首次更新前退出。没有270步正式结果。'],
    ['P2 · 未提交','Muse–SongBenchMean7–3e-4','纯Mean7需独立fail-closed adapter，不拿旧guard目标或Muse-SP结果替代。'],
    ['P3 · 后置','HeartMuLa → SongBloom / ACE','没有新增完整验证，不放旧head-only消融冒充主干结果。']
  ],runs,listening:listening.rows,
  audit_url:'https://github.com/zhanh-he/RL-paper-reading/tree/main/music-gen/reward-regime-study/weekly-2026-10-23'};
const destination=new URL('../site/demos/data/songbench-review.json',import.meta.url),text=JSON.stringify(value,null,2)+'\n';
if(process.argv.includes('--check'))assert.equal(fs.readFileSync(destination,'utf8'),text);else fs.writeFileSync(destination,text);
console.log('SongBench review: eight270-step arms, 5220 audio bindings, full validation; real completions and startup failures separated.');
