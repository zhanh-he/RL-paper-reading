import assert from 'node:assert/strict';
import fs from 'node:fs';
const root=new URL('../../music-gen/reward-regime-study/runs/2026-10-20-songbench-review/',import.meta.url);
const read=name=>JSON.parse(fs.readFileSync(new URL(name,root),'utf8'));
const audit=read('verified-recovery.json'),listening=read('listening.json');
const launch=read('../2026-10-20-songbench-priority/launch.json');
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
const museStep=launch.muse.latest['global_step/max_steps'];
const queueLabel=job=>`${job.job_id} · ${job.queue} · ${job.job_state==='Q'?'排队':job.job_state}`;
const job=reward=>launch.gadi.find(r=>r.reward===reward);
const value={updated_at:launch.checked_at,new_training_started:true,mean7_training_started:false,
  objective:'R = (Vocal + Instrumental + Melody + Structure + Arrangement + Mixing + Musicality) / 7; only SongBench, no extra penalty; KL separate.',
  objective_status:'User approved execution. Muse original-base SP/subhead controls first; pure Mean7 is queued/gated, not a completed result.',
  execution:`手动快照 ${launch.checked_at}；Muse-SP ${museStep}（5090）；两组Muse已提交Gadi；Diff Mean7等待Muse完成训练及验证；YuE2 Mean7已提交验收→正式任务。不是自动刷新或最终结果。`,
  base_identity:launch.base,
  launch_url:'https://github.com/zhanh-he/RL-paper-reading/blob/main/music-gen/reward-regime-study/runs/2026-10-20-songbench-priority/launch.json',
  scope:'Historical all-30 validation, not test or human MOS. Mean7 is a retrospective cross-score, NOT the historical optimized objective. Paired condition-bootstrap 95% intervals, one training seed, uncorrected multiple comparisons.',
  failure:'上一轮六个Muse/HeartMuLa作业在训练前因缺少generate_objective失败，未冒充完成。本轮已修复容器入口、单GPU同卡NCCL和codec/评分显存问题；Muse通过2次真实更新、首步KL=0、非零主干梯度、4条评分音频SHA及optimizer/RNG文件核验。YuE2 GPU验收尚未通过，不填正式结果。',
  priorities:[
    ['P0 · 正式运行','Muse–SP Richness–3e-4',`5090 · ${museStep}，目标270步；原始非GRPO checkpoint，新主干LoRA，beta0.05。补旧SP/单头减guard对比，非Mean7。`],
    ['P0 · 正式提交','Muse–SongBench-Instrument–3e-4',`${queueLabel(job('instrument'))}；同一原始Muse、全270训练/30验证，beta0.05，新主干LoRA。`],
    ['P0 · 正式提交','Muse–SongBench-Arrangement–3e-4',`${queueLabel(job('arrangement'))}；同一原始Muse、全270训练/30验证，beta0.05，新主干LoRA。`],
    ['P1 · 有限后续队列','DiffRhythm2–SongBenchMean7–3e-4','5090在Muse-SP训练和完整验证结束后启动；失败不抢跑。Miles主干LoRA，60s/G2/beta0.05/rank8/270步，新Mean7需重新验收。'],
    ['P1 · 已提交 / 验收门控','YuE2–SongBenchMean7–3e-4',`${queueLabel(job('mean7'))}；TRL导入/原生tokenizer检查已通过。真实2步→独立进程续到3步→全新270正式轮；任一失败即停。60s原生codec似然尚待GPU验收。`],
    ['P2','Muse–SongBenchMean7–3e-4','先完成上面的原始Muse三臂。纯Mean7不复用含guard的旧Muse reward，不伪装已经启动。'],
    ['P3','HeartMuLa → SongBloom / ACE','首轮不扩散模型和LR网格；这些模型不阻塞P1结果。']
  ],
  runs,listening:listening.rows,
  audit_url:'https://github.com/zhanh-he/RL-paper-reading/blob/main/music-gen/reward-regime-study/runs/2026-10-20-songbench-review/verified-recovery.json'};
const destination=new URL('../site/demos/data/songbench-review.json',import.meta.url);
const text=JSON.stringify(value,null,2)+'\n';
if(process.argv.includes('--check'))assert.equal(fs.readFileSync(destination,'utf8'),text);
else fs.writeFileSync(destination,text);
console.log(`SongBench review: historical six arms; Muse-SP ${museStep}; original-base Muse controls submitted; pure Mean7 queued/gated.`);
