import assert from 'node:assert/strict';
import fs from 'node:fs';
const source = new URL('../../music-gen/reward-regime-study/protocol.json', import.meta.url);
const target = new URL('../site/demos/data/reward-regime-protocol.json', import.meta.url);
const value = JSON.parse(fs.readFileSync(source));
assert.equal(value.training_authorized, true);
assert.equal(value.runs.length, 15);
assert.deepEqual(value.design.lr_grid, ['3e-4']);
assert.deepEqual(value.design.seeds, [42]);
assert.equal(value.framework_transition.ar_framework, 'TRL');
assert.equal(value.framework_transition.flow_framework, 'Miles-Diffusion');
assert.equal(value.framework_transition.parallel_flow_framework_comparison, false);
assert.equal(value.framework_transition.full_flow_lifecycle_accepted, true);
assert.equal(value.framework_transition.new_formal_backbone_training_authorized, true);
const flowAcceptance = JSON.parse(fs.readFileSync(new URL('../../music-gen/reward-regime-study/backbone-rl/diffrhythm2-miles-acceptance.json', import.meta.url)));
assert.equal(flowAcceptance.full_framework_e2e, true);
assert.equal(flowAcceptance.formal_training, false);
assert.deepEqual(flowAcceptance.before, flowAcceptance.after);
assert.equal(flowAcceptance.first.kl, 0);
assert.ok(flowAcceptance.second.kl > 0);
assert.ok([flowAcceptance.first, flowAcceptance.second].every(r => r.lr === 3e-4 && r.beta === .05));
assert.equal(value.framework_transition.public_page_url, 'https://zhanh-he.github.io/RL-paper-reading/demos/frameworks.html');
assert.ok(fs.existsSync(new URL('../site/demos/frameworks.html', import.meta.url)));
assert.equal(new Set(value.runs.map((r) => `${r.model}-${r.reward}`)).size, 15);
assert.ok(value.runs.every((r) => ['yue2', 'muse', 'heartmula', 'diffrhythm2', 'songbloom'].includes(r.model)));
assert.equal(value.rewards.length, 3);
assert.deepEqual(value.rewards.map((r) => r.id), ['richness', 'songbench', 'songbench-arrangement']);
assert.equal(value.reward_selection.third_reward, 'SongBench-Arrangement');
assert.ok(value.models.every((m) => m.mos === null));
assert.deepEqual(value.models.map((m) => m.id), ['yue2', 'heartmula', 'diffrhythm2', 'songbloom', 'muse', 'acestep', 'private']);
assert.equal(value.backbone_round.lr, 3e-4);
assert.equal(value.backbone_round.beta, .05);
assert.equal(value.backbone_round.scheduler, 'constant');
assert.deepEqual(value.backbone_round.priority, ['muse', 'diffrhythm2', 'heartmula', 'songbloom', 'acestep']);
assert.ok(value.runs.filter(r => r.model === 'muse').every(r => r.lr === 3e-4 && r.beta === .05 && r.scheduler === 'constant'));
assert.ok(value.models.every((m) => m.team_listening_estimate === null ||
  (Number.isFinite(m.team_listening_estimate) && m.team_listening_estimate >= 1 && m.team_listening_estimate <= 5)));
assert.match(value.expansion.listening_estimates, /not formal/);
assert.equal(value.models.find((m) => m.id === 'private').source, null);
assert.equal(value.expansion.new_training_submitted, true);
const progress = JSON.parse(fs.readFileSync(new URL('../../music-gen/reward-regime-study/progress.json', import.meta.url)));
const weekly=JSON.parse(fs.readFileSync(new URL('../../music-gen/reward-regime-study/weekly-2026-10-23/summary.json',import.meta.url)));
const muse=weekly.runs.find(r=>r.model==='muse'&&r.reward==='richness');
assert.equal(muse.updates,270);assert.equal(muse.binding.bound_audio_n,600);
value.status='20261023-muse-SP-complete-two-heads-startup-failed-diff-mean7-complete';
value.backbone_round.snapshot_at=weekly.updated_at;
value.backbone_round.report_url='https://github.com/zhanh-he/RL-paper-reading/tree/main/music-gen/reward-regime-study/weekly-2026-10-23';
value.backbone_round.status_by_model.muse='2026-10-23回收：原始非GRPO Muse-SP完成270步、600条音频绑定及0/270各30/30验证。Instrument/Arrangement因缺libpython3.11.so.1.0在scorer启动前失败，没有训练，不再排队。10.24s现有验证试听见本周汇报。';
value.backbone_round.status_by_model.yue2='历史SP/单项为已完成PyTorch轮。新增Mean7的TRL移植在gradient-checkpointing初始化时退出，0更新，未形成新正式结果。';
value.backbone_round.status_by_model.diffrhythm2='已核验旧SP/单项三组以及新增纯Mean7：均270步、0/50/100/270各30验证。Mean7终点5.116→5.055，未确立改善。新增曲线及试听见本周汇报。';
Object.assign(value.models.find(m=>m.id==='muse'),{status:'SP完成270步 / 另外两组启动失败',readiness:'measured',note:value.backbone_round.status_by_model.muse});
for(const arm of value.runs.filter(r=>r.model==='muse')){
  arm.status=arm.reward==='richness'?'complete':'failed-before-training';
  arm.verified_steps=arm.reward==='richness'?270:0;
  if(arm.reward==='richness')arm.full_valid_n=30;
  else arm.failure='libpython3.11.so.1.0 missing before scorer startup;0 optimizer updates';
}
value.execution_recovery={checked_at:weekly.updated_at,source:'weekly-2026-10-23/summary.json',scope:'Preregistered Oct20 plan enriched with actual Oct23 recovery; no new submission.'};
progress.runs.push({model:'muse',reward:'richness',steps:270,trace:muse.trace,complete:true,kl_unit:muse.kl_unit,
  scope:muse.scope,validation:muse.validation.map(v=>({step:v.step,n:30,mean:v.target_mean,receipt_sha256:v.receipt_sha256,
    cross_means:v.means,invalid_n:v.invalid_n}))});
progress.updated_at=weekly.updated_at;
const rendered = JSON.stringify(value, null, 2) + '\n';
if (process.argv.includes('--check')) assert.equal(fs.readFileSync(target, 'utf8'), rendered);
else fs.writeFileSync(target, rendered);
for (const name of ['5090', 'gadi']) {
  const archived = JSON.parse(fs.readFileSync(new URL(`../../music-gen/reward-regime-study/runs/2026-10-15-head-recovery-${name}/progress.json`, import.meta.url)));
  for (const run of archived.runs) {
    assert.equal(run.complete, false);
    assert.match(run.scope, /Head-only/);
    const arm=value.runs.find(r=>r.model===run.model&&r.reward===run.reward);
    assert.ok(arm); assert.equal(run.steps,arm.verified_steps);
    run.kl_unit=arm.kl_unit;
    if (run.model==='diffrhythm2') {
      for (const point of run.validation) {
        const path=`../../music-gen/reward-regime-study/runs/2026-10-15-head-recovery-5090/diffrhythm2-richness/valid_${String(point.step).padStart(6,'0')}.json`;
        const receipt=JSON.parse(fs.readFileSync(new URL(path,import.meta.url)));
        assert.equal(receipt.n,30);assert.equal(receipt.rows.length,30);
        point.cross_means=Object.fromEntries(['SP_Richness','Instrumental','Arrangement','Mixing','Musicality'].map(key=>[key,
          receipt.rows.reduce((sum,r)=>sum+(key==='SP_Richness'?r.richness.raw:r.songbench[key]),0)/30]));
        point.invalid_n=receipt.rows.filter(r=>r.guard.invalid).length;
      }
    }
    progress.runs.push(run);
  }
  if (archived.snapshot_at > progress.updated_at) progress.updated_at = archived.snapshot_at;
}
for (const run of progress.runs) {
  assert.equal(run.steps, run.trace.length);
  assert.ok(run.trace.every((p, i) => p.step === i+1 && Number.isFinite(p.reward) && Number.isFinite(p.kl)));
  assert.ok(run.validation.every((p) => p.n === 30 && Number.isFinite(p.mean) && p.receipt_sha256.length === 64));
}
const progressTarget = new URL('../site/demos/data/reward-regime-progress.json', import.meta.url);
const progressJSON = JSON.stringify(progress, null, 2)+'\n';
if (process.argv.includes('--check')) assert.equal(fs.readFileSync(progressTarget, 'utf8'), progressJSON);
else fs.writeFileSync(progressTarget, progressJSON);
console.log('SP/AI: 15 registered arms; completed, updating, sequential and preparation states distinguished; five fixed listening cases.');
const examples = JSON.parse(fs.readFileSync(new URL('../site/demos/data/reward-regime-examples.json', import.meta.url)));
assert.deepEqual(examples.display_indices, [0,10,20,30,31]);
assert.equal(examples.inputs.length, 32);
const [original, two, five] = [0,30,31].map(i => examples.inputs.find(r => r.index === i));
assert.equal(two.lyrics, original.lyrics); assert.equal(five.lyrics, original.lyrics);
assert.equal(two.seed, original.seed); assert.equal(five.seed, original.seed);
assert.equal(two.requested_count, 2); assert.equal(five.requested_count, 5);
assert.equal(two.original_style, original.style); assert.equal(five.original_style, original.style);
assert.match(two.scope, /excluded/); assert.match(five.scope, /excluded/);
const recoverySource=new URL('../../music-gen/reward-regime-study/recovery.json',import.meta.url);
if(fs.existsSync(recoverySource)) {
  const recovery=JSON.parse(fs.readFileSync(recoverySource));
  assert.equal(recovery.schema,1);
  assert.ok(recovery.audits.every((r)=>r.n===30));
  for(const receipt of recovery.validation_receipts) {
    assert.equal(receipt.n,30);assert.equal(receipt.rows.length,30);
    assert.ok(receipt.rows.every((r,i)=>r.index===i&&r.seed===5101+i&&r.source_audio_sha256.length===64));
    const measured=progress.runs.find((r)=>r.reward===receipt.reward).validation.find((r)=>r.step===receipt.step);
    const mean=receipt.rows.reduce((n,r)=>n+r.metrics.objective,0)/30;
    assert.ok(Math.abs(mean-measured.mean)<1e-10,'Full validation receipt/curve mismatch');
  }
  for(const c of recovery.badcases) {
    for(const [side,step] of [['baseline',0],['candidate',c.step]]) {
      const receipt=recovery.validation_receipts.find((r)=>r.reward===c.reward&&r.step===step);
      assert.equal(c[side].source_audio_sha256,receipt.rows[c.index].source_audio_sha256);
    }
    assert.ok(c.candidate.metrics.objective>c.baseline.metrics.objective);
  }
  for (const [id, record] of Object.entries(recovery.additional_models || {})) {
    assert.equal(id, 'diffrhythm2'); assert.equal(record.model, id);
    assert.equal(record.baseline.length, 3); assert.equal(record.long_replays.length, 6);
    assert.match(record.scope, /Deprecated head-only/);
    for (const row of [...record.baseline, ...record.long_replays]) {
      assert.ok([0,50,100].includes(row.step)); assert.ok([0,10,20].includes(row.index));
      assert.equal(row.seed, 5101+row.index);
      const path=`../../music-gen/reward-regime-study/runs/2026-10-15-head-recovery-5090/diffrhythm2-richness/valid_${String(row.step).padStart(6,'0')}.json`;
      const receipt=JSON.parse(fs.readFileSync(new URL(path,import.meta.url)));
      assert.equal(receipt.n,30); assert.equal(receipt.rows.length,30);
      const source=receipt.rows[row.index];
      assert.equal(row.asset.source_audio_sha256, source.audio_sha256);
      assert.ok(Math.abs(row.asset.metrics.objective-source.mean)<1e-10);
    }
  }
  const recoveryTarget=new URL('../site/demos/data/reward-regime-recovery.json',import.meta.url);
  const encoded=JSON.stringify(recovery,null,2)+'\n';
  if(process.argv.includes('--check'))assert.equal(fs.readFileSync(recoveryTarget,'utf8'),encoded);
  else fs.writeFileSync(recoveryTarget,encoded);
}
