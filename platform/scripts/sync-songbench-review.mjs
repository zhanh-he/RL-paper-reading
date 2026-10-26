import assert from 'node:assert/strict';
import fs from 'node:fs';
const study=new URL('../../music-gen/reward-regime-study/',import.meta.url);
const read=name=>JSON.parse(fs.readFileSync(new URL(name,study),'utf8'));
const weekly=read('weekly-2026-10-23/summary.json'),listening=read('runs/2026-10-20-songbench-review/listening.json');
const launch=read('runs/2026-10-20-songbench-priority/launch.json');
const checked=new Date(weekly.updated_at).toLocaleString('zh-CN',{timeZone:'Australia/Perth',hour12:false})+' AWST';
assert.equal(weekly.runs.length,10);
const runs=weekly.runs.map(r=>{
  assert.equal(r.updates,270);assert.equal(r.trace.length,270);assert.ok(r.validation.every(v=>v.n===30));
  return {id:r.id,model:r.model,reward:r.reward,experiment:r.experiment,updates:r.updates,trace:r.trace,kl_unit:r.kl_unit,
    validation:r.validation.map(v=>({step:v.step,n:v.n,target_mean:v.target_mean,means:v.means,invalid_n:v.invalid_n})),
    mean7_delta:r.mean7.delta,mean7_delta_ci95:r.mean7.ci95,binding:r.binding,
    experiment_sha256:r.experiment_sha256,steps_sha256:r.steps_sha256,
    validation_receipt_sha256:r.validation.map(v=>({step:v.step,sha256:v.receipt_sha256}))};
});
assert.equal(runs.reduce((n,r)=>n+r.binding.bound_audio_n,0),6600);assert.equal(listening.rows.length,20);
const value={updated_at:weekly.updated_at,new_training_started:true,mean7_training_started:true,
  objective:'R = (Vocal + Instrumental + Melody + Structure + Arrangement + Mixing + Musicality) / 7; only SongBench, no extra penalty; KL separate.',
  objective_status:'Muse SP-v2/Mean7 LR3e-4 and Diff Mean7 LR3e-4/1e-4 verified270/all30. Muse Mean7 LR1e-4 training complete, validation failed. YuE2 Mean7 failed native NAR acceptance.',
  execution:`手动回收 ${checked}：10个完整SP/AI实验、6600音频绑定。新增Muse Mean7、SP v2与Diff低LR；四个checkpoint均30/30。训练完成与验证失败分开，详细曲线与80个试听绑定见本周汇报。`,
  base_identity:launch.base,
  launch_url:'https://github.com/zhanh-he/RL-paper-reading/tree/main/music-gen/reward-regime-study/weekly-2026-10-23',
  scope:weekly.method.scope,
  failure:'Muse旧Instrument/Arrangement启动失败；新Mean7 LR1e-4已训练270步，但验证因GCC版本导致FlashInfer JIT失败。YuE2 Mean7 v2在NAR/PEFT路径缺rotary_emb，0更新。Muse test遇到无AUDIO token输出后评估失败，不能把遗留子进程当作正常进度。失败不等于模型不能RL。',
  priorities:[
    ['P0 · 完成并核验','Muse–SP Richness–3e-4 · v2','270步；0/50/100/270各30/30；同一原始非GRPO起点，beta0.05，10.24s codec；旧轮另存。'],
    ['P0 · 启动失败','Muse–SongBench-Instrument / Arrangement–3e-4','181718185 /181718187：缺少libpython3.11.so.1.0；没有optimizer更新。不再标排队。'],
    ['P1 · 完成并核验','DiffRhythm2–SongBenchMean7–3e-4 /1e-4','各270步、120/120验证；60s预算，Miles主干LoRA。两个终点均约5.055，未确立改善。'],
    ['P1 · 验收失败','YuE2–SongBenchMean7–3e-4','182083826：H200首次采样NAR/PEFT缺rotary_emb，0更新；历史PyTorch结果不改写。'],
    ['P0 · 完成并核验','Muse–SongBenchMean7–3e-4','270步、120/120验证；Mean7 3.971→4.318。1e-4完成训练但验证失败；全量train与test状态详见本周汇报。'],
    ['P3 · 后置','HeartMuLa → SongBloom / ACE','没有新增完整验证，不放旧head-only消融冒充主干结果。']
  ],runs,listening:listening.rows,
  audit_url:'https://github.com/zhanh-he/RL-paper-reading/tree/main/music-gen/reward-regime-study/weekly-2026-10-23'};
const destination=new URL('../site/demos/data/songbench-review.json',import.meta.url),text=JSON.stringify(value,null,2)+'\n';
if(process.argv.includes('--check'))assert.equal(fs.readFileSync(destination,'utf8'),text);else fs.writeFileSync(destination,text);
console.log('SongBench review: ten270-step arms,6600 audio bindings; full validation separated from training-only and failed evaluation.');
