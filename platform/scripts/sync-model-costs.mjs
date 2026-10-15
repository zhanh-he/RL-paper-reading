import fs from 'node:fs';
import assert from 'node:assert/strict';

const root = new URL('../../music-gen/lyrics2song/model-costs/', import.meta.url);
const read = (name) => JSON.parse(fs.readFileSync(new URL(name, root)));
const optional = (name) => fs.existsSync(new URL(name, root)) ? read(name) : null;
const hf = read('native-hf-final-counts.json').components;
const heart = read('native-heart-counts.json').components;
const bloom = read('native-songbloom-counts.json').components;
const weights = read('weight-audit.json').components;
const native = { ...hf, ...heart, ...bloom,
  'diffrhythm2-core': { unique_parameters: 1136249664, lora: weights['diffrhythm2-core'].lora_shape_plan } };
const local = read('timing-5090.json').measurements;
const historical = read('timing-heart.json').measurements;
const codec = optional('mucodec-parameters.json');
const median = (values) => {
  const sorted = [...values].sort((a,b) => a-b), half = Math.floor(sorted.length/2);
  return sorted.length % 2 ? sorted[half] : (sorted[half-1]+sorted[half])/2;
};
const records = [
  ['yue2','YuE2','YuE2-3B · AR/NAR MoT','yue2-core',
    '新计划：仅 semantic AR attention+MLP；NAR / VAE 固定。TRL迁移尚未提交。旧 q/v rank4 为802,816参数，不能替代此新计划数值。',
    'VAE 132,616,130；完整 MoT+VAE 3,712,968,706。固定位置编码50,331,648不计参数。',
    'https://huggingface.co/m-a-p/YuE2-3B'],
  ['muse','Muse','Muse-0.6b · Qwen3','muse-core',
    '新正式任务：rank8 all-linear attention+MLP；tied embedding / output head只计一次，codec固定。',
    codec ? `MuCodec ${codec.unique_parameters.toLocaleString('en-US')}；LM+已加载codec ${(native['muse-core'].unique_parameters+codec.unique_parameters).toLocaleString('en-US')}。` : 'MuCodec独立列项，尚未取得原生加载参数回执；不能把0.6126B当整链总数。',
    'https://huggingface.co/bolshyC/Muse-0.6b'],
  ['diffrhythm2','DiffRhythm2','ASLP-lab v2 · DiT','diffrhythm2-core',
    '已验收并启动：112个DiT attention+MLP模块；rank8 / alpha16。',
    '另有Music VAE decoder和MuQ-MuLan条件编码器；不把它们算进1.1362B主干，也不冒称已核完整整链总数。',
    'https://huggingface.co/ASLP-lab/DiffRhythm2'],
  ['heartmula','HeartMuLa','oss-3B-happy-new-year','heartmula-core',
    '新正式任务：217个global/local attention+MLP模块，8个codebook参与loss；codec固定。',
    'HeartCodec 1,655,308,877；完整LM+codec 5,593,422,413。含codec内flow网络，固定buffer不计。',
    'https://huggingface.co/HeartMuLa/HeartMuLa-oss-3B-happy-new-year'],
  ['songbloom','SongBloom','社区恢复 full-150s · 36 AR / 12 NAR','songbloom-core',
    '形状核验的新计划：60个nar_dit线性模块；AR planner / VAE固定；完整Miles worker尚未提交。',
    'VAE 168,695,426；去重整链 2,041,068,355。原checkpoint重复VAE别名与38,064个buffer不计。',
    'https://arxiv.org/abs/2506.07634'],
  ['acestep15','ACE-Step1.5','v15-base · 非XL / 非Turbo','acestep-base',
    '形状核验的新计划：decoder self/cross-attention+MLP，264模块；尚未提交RL，不借用Turbo测速。',
    'Text encoder 595,776,512 + VAE168,695,426：无planner整链3,158,344,456；可选1.7B planner实为1,854,243,840，总计5,012,588,296。',
    'https://huggingface.co/ACE-Step/acestep-v15-base'],
];
const models = records.map(([id,name,checkpoint,key,plan,components,url]) => {
  const raw = native[key], parameters = raw.unique_parameters, lora = raw.lora.parameters;
  assert.ok(Number.isSafeInteger(parameters) && parameters>0);
  assert.ok(Number.isSafeInteger(lora) && lora>0 && raw.lora.modules>0);
  return { id,name,checkpoint,parameters,lora_parameters:lora,lora_modules:raw.lora.modules,
    lora_rank:8,lora_alpha:16,lora_percent:100*lora/parameters,plan,components,
    parameter_evidence:key==='diffrhythm2-core' ? 'Native loaded-model count + safetensors shape cross-check' : 'Native unique nn.Parameters; meta construction, explicit HF tie_weights; fixed buffers excluded',
    source:url };
});
const timing = [];
for (const id of ['yue2','muse','acestep15']) {
  const raw = optional(`${id==='acestep15'?'acestep':id}-timing.json`);
  const rows = (raw?.rows || []).filter(r => !r.warmup && [60,300].includes(r.target_seconds));
  timing.push({ id,kind:rows.length?'measured':'unavailable',hardware:raw?.device || 'n/a · 未取得有效计时',
    backend:raw?.backend || '尚无本checkpoint的完整推理计时',scope:raw?.scope || '不以训练更新耗时或Turbo宣传速度填充Base的RTF。',
    n:rows.length,n_label:rows.length?`${rows.length}次计时 / 1条条件`:null,rows,load_seconds:raw?.model_load_seconds ?? null,duration_control:raw?.duration_control ?? null,
    source:rows.length?`${id==='acestep15'?'acestep':id}-timing.json`:null });
}
for (const [id,raw] of [['diffrhythm2',local.diffrhythm2],['heartmula',historical.heartmula]]) {
  const complete=raw.rows.filter(r=>Math.abs(r.seconds-60)<=.1);
  assert.ok(complete.length>0);
  const rate = median(complete.map(r=>r.generation_seconds/r.seconds));
  timing.push({id,kind:id==='diffrhythm2'?'measured-rollout-proxy':raw.kind,hardware:raw.device,
    backend:id==='diffrhythm2'?'训练worker · FP16 /SDPA /block KV cache /32时间点 /CFG双路，含搬运与落盘':'BF16 LM + FP32 HeartCodec · 历史rollout',
    scope:raw.scope+`. Full receipt has${raw.rows.length} rows, including${raw.rows.length-complete.length} early stops. The60s statistics and300s extrapolation use only${complete.length} outputs within60±0.1s; early stops are not padded or relabeled60s.`+
      (id==='diffrhythm2'?' NativeWorker timer includes offload/device transfers, first condition encode/cache, decode, FLAC save and cleanup. This is not an optimized resident-GPU pure-inference benchmark. Published loader disables FlexAttention despite checkpoint config=true; actual path is SDPA with block KV cache. Stage-specific seconds were not measured.':''),
    n:complete.length,n_label:`${complete.length}/${raw.rows.length}条约60s · 中位数`,median_rtf:rate,
    measured_audio_range:[Math.min(...raw.rows.map(r=>r.seconds)),Math.max(...raw.rows.map(r=>r.seconds))],
    rtf_range:[Math.min(...raw.rows.map(r=>r.generation_seconds/r.seconds)),Math.max(...raw.rows.map(r=>r.generation_seconds/r.seconds))],
    rows:[{target_seconds:60,actual_audio_seconds:median(complete.map(r=>r.seconds)),wall_seconds:median(complete.map(r=>r.generation_seconds)),rtf:rate,
           kind:'measured',caveat:`${complete.length}条足长输出；${raw.rows.length-complete.length}条早停不混入60s统计`},
          {target_seconds:300,wall_seconds:rate*300,rtf:rate,kind:'linear-estimate',
           caveat:id==='heartmula'?'超默认240s，尚未验证原生300s输出':'未执行300s，长上下文可能非线性变慢'}],
    source:id==='diffrhythm2'?'timing-5090.json':'timing-heart.json'});
}
timing.push({id:'songbloom',kind:'author-reference',hardware:'作者未注明测速GPU',backend:'论文 full / 36 diffusion steps，不是本站8步smoke',n:null,
  median_rtf:1.649,scope:'论文Table2报告RTF1.649，未明确测速硬件和时段；不可作本站跨硬件排行。60s是按该比率换算，不是本站实测。',
  rows:[{target_seconds:60,wall_seconds:1.649*60,rtf:1.649,kind:'author-rate-estimate'},
        {target_seconds:300,wall_seconds:null,kind:'unsupported',caveat:'本checkpoint上限150s，不把拼接或线性算术伪装成原生300s'}],
  source:'https://arxiv.org/html/2506.07634v1#S6'});
const localBloom=optional('songbloom-timing.json');
if(localBloom){
  const row=timing.find(r=>r.id==='songbloom');
  const measured=localBloom.rows.filter(r=>!r.warmup&&r.target_seconds===60);
  assert.equal(measured.length,1);
  Object.assign(row,{kind:'measured',hardware:localBloom.device,backend:localBloom.backend,n:1,n_label:'1次计时 / 1条条件',
    median_rtf:undefined,scope:localBloom.scope,rows:[...measured,row.rows[1]],
    load_seconds:localBloom.model_load_seconds,duration_control:localBloom.duration_control,source:'songbloom-timing.json'});
}
timing.sort((a,b)=>models.findIndex(r=>r.id===a.id)-models.findIndex(r=>r.id===b.id));
for(const entry of read('published-speed-sources.json').entries.filter(e=>e.selected!==false)){
  const rate=entry.normalization==='reciprocal'?1/entry.reported_rtf:
    entry.normalization==='wall_div_audio'?entry.reported_wall_seconds/entry.reported_audio_seconds:entry.reported_rtf;
  assert.ok(Number.isFinite(rate)&&rate>0);
  const unverified=entry.publish_duration_estimates===false;
  timing.push({...entry,kind:'author-reference',median_rtf:unverified?null:rate,claim_rtf:unverified?rate:undefined,n:null,
    rows:[60,300].map(seconds=>unverified?
      {target_seconds:seconds,wall_seconds:null,kind:'boundary-unverified',caveat:entry.duration_caveat}:
      entry.max_output_seconds!==null&&seconds>entry.max_output_seconds?
      {target_seconds:seconds,wall_seconds:null,kind:'unsupported',arithmetic_only_seconds:rate*seconds,
        caveat:`官方上限${entry.max_output_seconds}s；RTF算术外推${Math.round(rate*seconds)}s不代表可生成该长度`}:
      {target_seconds:seconds,wall_seconds:rate*seconds,rtf:rate,kind:'author-rate-estimate',caveat:entry.duration_caveat})});
}
const acceleration=read('vllm-support.json');
acceleration.projections=[];
for(const id of ['yue2','muse']){
  const raw=read(`${id}-timing.json`);
  for(const row of raw.rows.filter(r=>!r.warmup)){
    const ar=id==='yue2'?row.timing.abc.seconds+row.timing.semantic.seconds:row.lm_seconds;
    const other=row.wall_seconds-ar;
    assert.ok(other>0&&ar>0);
    acceleration.projections.push({id,name:models.find(m=>m.id===id).name,
      target_seconds:row.target_seconds,hardware:raw.device,baseline_seconds:row.wall_seconds,
      ar_seconds:ar,unchanged_seconds:other,ar_fraction:ar/row.wall_seconds,
      scenarios:[2,4].map(factor=>({assumed_ar_speedup:factor,estimated_wall_seconds:ar/factor+other,
        estimated_e2e_speedup:row.wall_seconds/(ar/factor+other)})),
      ar_only_infinite_speedup_bound:row.wall_seconds/other,
      scope:'条件估算，非vLLM预测或实测。假设AR阶段2×/4×，其他阶段不变；AR无限快的上界也只对这个假设成立。YuE2-Omni还可能优化NAR，因此不是其完整后端上界。'});
  }
}
const vllmMuse=optional('muse-vllm-timing.json');
if(vllmMuse?.rows.some(r=>!r.warmup)){
  assert.equal(vllmMuse.device,read('muse-timing.json').device,'Do not present a cross-hardware ratio as backend speedup');
  const arm=acceleration.models.find(m=>m.id==='muse');
  arm.measured_rows=vllmMuse.rows.filter(r=>!r.warmup).map(row=>{
    const base=read('muse-timing.json').rows.find(r=>r.target_seconds===row.target_seconds);
    return {...row,e2e_speedup:base.rtf/row.rtf,ar_speedup:base.lm_seconds/row.lm_seconds};
  });
  arm.speedup_status='端到端配对实测 · 相同型号A100 / 各长度单次，非交错重复';
  arm.support=vllmMuse.backend;arm.measurement_scope=vllmMuse.scope;
  arm.receipt='muse-vllm-timing.json';
}
const value={schema:1,checked_at:'2026-10-15',title:'模型规模与推理等待时间',
  definition:'RTF = wall-clock generation seconds / actual output audio seconds; lower is faster. 1B=1e9,1M=1e6. Parameters are unique nn.Parameters, not bytes or every checkpoint tensor.',
  scope:'Six exact local parameter/LoRA audits and local batch1 timing evidence, plus three selected published speed references (LeVo2-medium only; large archived). Backend compatibility, measured acceleration and explicitly conditional AR-only estimates are separated. Published batch sizes/timing boundaries are not assumed identical to ours. No training interruption. Not a speed leaderboard.',
  interpretation:'实测60s不等于实测300s。估算只按同口径RTF线性外推，忽略固定开销和长度非线性。模型加载、排队、下载、reward评分和训练更新不包含在热推理等待时间内；各回执注明额外范围。',
  models,timing,acceleration,timing_audit:read('inference-boundaries.json'),
  references:[{label:'YuE2参数口径与MoT',url:'https://arxiv.org/html/2609.33757v1#A1.SS5'},
    {label:'DiffRhythm2官方4090测速：120s /25.6s /32steps；非本站5090',url:'https://arxiv.org/html/2510.22950v2#A2'},
    {label:'HeartMuLa官方RTF≈1.0，硬件未给，不代替本站记录',url:'https://github.com/HeartMuLa/heartlib#-todos'},
    {label:'SongBloom原生150s上限与RTF',url:'https://arxiv.org/html/2506.07634v1#S5.SS2'},
    {label:'ACE-Step1.5区分base与turbo；不从“每首<2s”虚构RTF',url:'https://github.com/ace-step/ACE-Step-1.5'}]};
const output=new URL('../site/demos/data/model-costs.json',import.meta.url);
const serialized=JSON.stringify(value,null,2)+'\n';
if(process.argv.includes('--check'))assert.equal(fs.readFileSync(output,'utf8'),serialized);
else fs.writeFileSync(output,serialized);
console.log('Six-model parameter/LoRA counts and latency evidence verified; direct timing, extrapolation, author reference and unsupported durations separated.');
