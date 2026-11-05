"""Sanitized, SHA-bound audit of actual prompt/CFG generations. No human gold.

Private input files GROUP_{excerpts,auditory,roughness,signal}.json are joined by
excerpt IDs AND exact WAV SHA256. Public JSON contains only an explicit allowlist.
All windows are prescribed in the excerpt manifest, never selected by a score.
"""
import argparse,collections,csv,hashlib,io,json,math,statistics,re
from pathlib import Path

BASE='https://github.com/zhanh-he/RL-paper-reading/blob/main/music-gen/reward-regime-study/naturalness/'
PUBLIC_AUDIO='./audio/artifact-control-20261105/'
METRICS=['modulation_depth','roughness70','roughness70_p95','resonance_protected','suno_descriptor_distance']
ARM_LABELS={'base':'原始提示','literal_zh':'追加中文笼统否定','literal_en':'追加英文笼统否定','concrete':'具体发声描述＋保留配器','cfg0':'native CFG0','cfg1':'native CFG1','audio_reference':'共享Suno音频参考'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def finite(v):return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
def percentile(xs,p):
 xs=sorted(xs);q=(len(xs)-1)*p;i=int(q);j=min(i+1,len(xs)-1);return xs[i]*(j-q)+xs[j]*(q-i) if i!=j else xs[i]
def avg(xs):
 xs=[x for x in xs if finite(x)];return statistics.mean(xs) if xs else None
def number(v,n=4):return f'{v:.{n}f}' if finite(v) else '未运行'
def relative(a,b):return (b-a)/abs(a) if finite(a) and finite(b) and abs(a)>1e-12 else None
def transform(p):return [p['modulation_depth'],math.log(p['roughness70']+1e-4),math.log(p['resonance_protected']+1e-4)]
def label_model(case_id):return 'SongBloom（社区恢复full150s）' if case_id.startswith('bloom_') else 'Muse0.6B＋MuCodec' if case_id.startswith('muse_') else 'DiffRhythm2'
def proto_config(suno):
 vectors=[transform(s['psycho']) for s in suno['samples'] if s['psycho'].get('valid')]
 centers=[statistics.median([v[i] for v in vectors]) for i in range(3)]
 scale=[max(percentile([v[i] for v in vectors],.75)-percentile([v[i] for v in vectors],.25),1e-6) for i in range(3)]
 return dict(features=['modulation_depth','log(roughness70+1e-4)','log(resonance_protected+1e-4)'],center=centers,iqr=scale,reference_clips=len(vectors),reference_songs=len({r['song_id'] for r in suno['samples']}),reference_families=len({r['source_family'] for r in suno['samples']}),formula='sqrt(mean(((transform(feature)-Suno_median)/Suno_IQR)^2))',direction='Lower means closer to the center of these three descriptors in this small Suno sample, NOT greater perceptual Suno likeness or fewer artifacts.',caveat='48 excerpts are clustered in16songs/8families; source/style/content/decoder confounds remain. No source classifier or calibrated probability is trained.')
def distance(p,cal):
 v=transform(p);return math.sqrt(statistics.mean([((x-m)/s)**2 for x,m,s in zip(v,cal['center'],cal['iqr'])]))
def modulation_trace(g):
 maps=g['maps'];depth=maps['modulation_depth'];energy=maps['band_energy_relative'];n=len(maps['time_s']);values=[]
 for i in range(n):
  w=[energy[j][i]**2 if energy[j][i]>.001 else 0 for j in range(len(energy))];den=sum(w)
  values.append(sum(depth[j][i]*w[j] for j in range(len(w)))/max(den,1e-20))
 return dict(time_s=maps['time_s'],value=values)
def clean_track(row,profile,group):
 # Never copy a whole private row/profile. Paths, prompts from private pilots and
 # token files are deliberately absent from the public allowlist.
 fields=['id','family','arm','seed','sample_rate','channels','duration_s','actual_duration_s','audio_sha256','audio_pcm_sha256','latent_sha256','initial_noise_sha256','cfg_strength','standard_cfg_scale','decoder_seed','cfg_coef','reference_id','reference_sha256','guidance_scale','actual_guidance_scale','guidance_source_route','guidance_route','guidance_passed_to_model','guidance_calls','effective_guidance_scales','analysis_include','source_case_id','audio_tokens_sha256','audio_token_count','exact_token_control_group','same_decoder_random_draws_as_first_identical_codes','decoder_random_draws','contrast','ode_seconds','decode_seconds','end_to_end_seconds','generation_seconds','inference_seconds','gpu_peak_allocated_bytes','rms_dbfs','peak','out_of_range_fraction','full_mp3_sha256','keep_for_listening']
 d={k:row[k] for k in fields if k in row};d.update(model=label_model(row['id']),source_group=group,full_audio=PUBLIC_AUDIO+row['id']+'__full.mp3',human_rating=None)
 actual=row.get('effective_guidance_scales')
 if actual and all(finite(x) for x in actual) and len(set(actual))==1:d['actual_guidance_scale']=actual[0]
 # These are newly authored public experiment prompts, not the old internal pilots.
 if 'style' in row and not str(row['style']).startswith('audio_reference:'):d['prompt']=row['style']
 if row.get('arm')=='audio_reference':d['prompt']='Same permitted10s Suno reference audio plus newly authored experiment lyrics; no invented free-text instruction interface.'
 return d

def build(a):
 w=Path(a.workspace);suno=read(a.suno_audit);suno_by={r['id']:r for r in suno['samples']};cal=proto_config(suno)
 model_provenance=[]
 for filename in ['diff_run.json','diff_reference_run.json','bloom_run.json','muse_run.json']:
  path=w/filename
  if not path.exists():continue
  pp=read(path)['profile'];keep=['model','assets','asset_repository','asset_revision','asset_source_status','checkpoint_compatibility','device','torch','seconds','steps','num_steps','native_requested_seconds','settings','load_seconds','cfg_convention','seed_protocol','condition_interface','inference','scope','decoder_weight_sha256','native_window_duration','seed']
  item={k:pp[k] for k in keep if k in pp};item['record_id']=filename.removesuffix('.json');item['private_receipt_sha256']=sha(path)
  if filename.startswith('bloom'):item['verification_note']='Community-restored checkpoint, not independently matched to inaccessible author endpoint; original academic-only restriction retained.'
  if filename.startswith('muse'):item['verification_note']='Local Muse/MuCodec cache and public API instrumentation; actual used guidance is captured separately.'
  model_provenance.append(item)
 manifest_files={};rows=[];tracks={};profiles={};all_audio={};all_rough={};all_signals={};input_sha={};status=[]
 aq={}
 aqpath=w/a.audiobox
 for file in [aqpath,w/'muse_audiobox_scores.jsonl']:
  if not file.exists():continue
  for r in [json.loads(line) for line in file.read_text().splitlines() if line.strip()]:
   if r['id'] in aq and (aq[r['id']]['audio_sha256']!=r['audio_sha256'] or aq[r['id']]['scores']!=r['scores']):raise ValueError('Ambiguous Audiobox duplicate '+r['id'])
   aq[r['id']]=r
 muse_batch=read(w/'muse_batch.json') if (w/'muse_batch.json').exists() else {'cases':[]};muse_prompts={r['id']:r for r in muse_batch['cases']}
 for group in a.groups:
  required={kind:w/f'{group}_{kind}.json' for kind in ['excerpts','auditory','roughness']}
  missing=[str(p.name) for p in required.values() if not p.exists()]
  if missing:
   if a.skip_missing:status.append(dict(group=group,status='not_included_missing_inputs',missing=missing));continue
   raise FileNotFoundError(', '.join(missing))
  m,au,rr=[read(required[k]) for k in ['excerpts','auditory','roughness']]
  if au['status']!='complete' or rr['status']!='complete':raise ValueError(group+' scoring incomplete')
  maps={r['id']:r for r in au['samples']};rough={r['id']:r for r in rr['samples']}
  signalfile=w/f'{group}_signal.json';sig=read(signalfile) if signalfile.exists() else None
  signalmap={r['id']:r for r in sig.get('rows',sig.get('samples',[]))} if sig else {}
  profiles[group]={'auditory':au['profile'],'roughness':rr['profile']};input_sha[group]={k:sha(v) for k,v in required.items()};manifest_files[group]=m
  for t in m['tracks']:
   if t['id'] in tracks:raise ValueError('Duplicate track ID')
   if t.get('analysis_include') is False:continue
   tracks[t['id']]=clean_track(t,{},group)
   if t['id'].startswith('muse_'):
    source=t.get('source_case_id');prompt=muse_prompts.get(source,{})
    if prompt:tracks[t['id']]['prompt']=prompt.get('global_style','')+' / segment description: '+prompt.get('segment_description','')
  for mrow in m['samples']:
   rid=mrow['id']
   if mrow['source_case_id'] not in tracks:continue
   case=tracks[mrow['source_case_id']]
   if mrow.get('analysis_include') is False or mrow.get('keep_for_listening') is False or case.get('keep_for_listening') is False or str(case.get('arm','')).startswith('api_noop'):continue
   g=maps[rid]['auditory'];r=rough[rid];wave_sha=mrow['excerpt_sha256']
   if maps[rid]['audio_sha256']!=wave_sha or r['audio_sha256']!=wave_sha:raise ValueError('Scoring WAV hash mismatch '+rid)
   sg=signalmap.get(rid)
   if sg and sg['audio_sha256']!=wave_sha:raise ValueError('Signal hash mismatch '+rid)
   q=aq.get(rid)
   if q and q['audio_sha256']!=wave_sha:raise ValueError('Audiobox hash mismatch '+rid)
   valid=bool(g.get('valid') and r.get('valid'));asper=r.get('asper',{})
   p=dict(valid=valid,modulation_depth=g.get('modulation_depth_20_300') if valid else None,modulation_coherent=g.get('coherent_modulation_depth') if valid else None,modulation_peak_hz=g.get('fast_modulation_peak_hz') if valid else None,roughness70=asper.get('mean'),roughness70_p95=asper.get('p95'),roughness60=None,roughness80=None,roughness_raw=None,resonance_raw=g.get('isolated_resonance_proxy') if valid else None,resonance_3_8=g.get('resonance_3_8_proxy') if valid else None,resonance_protected=g.get('harmonic_veto_resonance_proxy') if valid else None,resonance_frequency_hz=g.get('harmonic_veto_candidate_hz') if valid else None)
   p['suno_descriptor_distance']=distance(p,cal) if valid else None
   timeline=dict(modulation=modulation_trace(g),roughness=dict(time_s=r.get('time_s',[]),value=r.get('time_asper',[])),resonance=dict(time_s=g['resonance_time']['time_s'],value=g['resonance_time']['harmonic_veto_excess'])) if valid else None
   cfg=case.get('cfg_strength');pc=dict(family=case.get('family'),arm=case.get('arm'),prompt=case.get('prompt',''),seed=case.get('seed'),native_cfg_strength=cfg,reference_id=case.get('reference_id'),guidance_scale=case.get('guidance_scale'),actual_guidance_scale=case.get('actual_guidance_scale'),guidance_route=case.get('guidance_route',case.get('guidance_source_route')))
   rights='Newly authored lyrics and prompts; original model generation, academic research demo. No raw training audio or weights redistributed.'
   if rid.startswith('bloom_'):rights+=' SongBloom checkpoint is community-restored; archived original license limits model use to academic purposes. Community Apache card does not override that restriction; no unrestricted output-rights claim.'
   if case.get('reference_id'):rights+=' Conditioning clip is from the Muse publisher-declared MIT Suno dataset; separate reference SHA is retained.'
   variant_label=ARM_LABELS.get(case.get('arm'),case.get('arm',''))
   if case['id'].startswith('muse_') and case.get('actual_guidance_scale') is not None:variant_label+=f" · 实际decoder guidance{case['actual_guidance_scale']:g}"
   row=dict(id=rid,label=f"{label_model(case['id'])} · {case.get('family')} · {variant_label} · seed{case.get('seed')} · {mrow['excerpt_start_s']:.2f}–{mrow['excerpt_start_s']+10:.2f}s",kind='actual_generated_control',model=label_model(case['id']),checkpoint='Frozen pretrained inference; no RL update',input_id=f"{case['source_group']}/{case.get('family')}/s{case.get('seed')}",source_family=f"{case['source_group']}/{case.get('family')}/ownlyrics-v1",pair_group=mrow['pair_group'],same_input_verified=False,source_case_id=case['id'],source_original_sha256=mrow['source_original_sha256'],excerpt_sha256=wave_sha,mp3_sha256=mrow['mp3_sha256'],audio=PUBLIC_AUDIO+rid+'.mp3',full_audio=PUBLIC_AUDIO+case['id']+'__full.mp3',full_mp3_sha256=mrow['full_mp3_sha256'],duration_s=10.,sample_rate=48000,channels=mrow['channels'],excerpt_start_s=mrow['excerpt_start_s'],source_duration_s=mrow['source_duration_s'],window_index=mrow['window_index'],transform='none',selection_reason=mrow['selection_reason'],rights=rights,human_rating=None,psycho=p,prompt_control=pc,timeline=timeline,signal=sg['signal'] if sg else {},audiobox=q['scores'] if q else None)
   row['source_url']='https://github.com/xiaomi-research/diffrhythm2' if rid.startswith('diff_') else 'https://huggingface.co/rsxdalv/SongBloom' if rid.startswith('bloom_') else 'https://github.com/yuhui1038/Muse'
   rows.append(row);all_audio[rid]=maps[rid];all_rough[rid]=r
  status.append(dict(group=group,status='included_complete_scores',tracks=len(m['tracks']),clips=sum(r['source_case_id'] in {t['id'] for t in m['tracks']} for r in rows)))
 if not rows:raise ValueError('No complete, real, scored controls')
 by={r['id']:r for r in rows};case_clips=collections.defaultdict(list)
 for r in rows:case_clips[r['source_case_id']].append(r)
 for cid,t in tracks.items():
  clips=case_clips[cid];t['scored_windows']=[dict(id=r['id'],start_s=r['excerpt_start_s'],end_s=r['excerpt_start_s']+10) for r in clips]
  t['window_mean']={k:avg([r['psycho'][k] for r in clips]) for k in METRICS};t['audiobox_window_mean_PQ']=avg([r['audiobox'].get('PQ') for r in clips if r.get('audiobox')]);t['mean_scope']='Mean of prescribed10s windows, NOT a whole-song quality score.'
 # A fixed, compact playlist; every requested arm remains in the full sample list.
 pairs=[]
 def first_clip(cid):return min(case_clips.get(cid,[]),key=lambda r:r['window_index']) if case_clips.get(cid) else None
 def pair(pid,label,ca,cb,purpose,caveat,ref=None):
  aa,bb=first_clip(ca),first_clip(cb)
  if not aa or not bb:return
  obs='；'.join([f"{title} {number(aa['psycho'][key])}/{number(bb['psycho'][key])}" for key,title in [('modulation_depth','调制'),('roughness70','DW70'),('resonance_protected','保护共振')]])
  pc=dict(id=pid,label=label,a=aa['id'],b=bb['id'],purpose=purpose,observation=obs,caveat=caveat,checkpoints=[])
  for r in [aa,bb]:
   for method in ['modulation','roughness','resonance']:
    ss=(r.get('timeline') or {}).get(method)
    if not ss or not ss['value'] or max(ss['value'])<=0:continue
    i=max(range(len(ss['value'])),key=lambda i:ss['value'][i]);pc['checkpoints'].append(dict(sample_id=r['id'],clip_s=ss['time_s'][i],source_s=r['excerpt_start_s']+ss['time_s'][i],method=method,note='Nominal analysis-window center; not an artifact label.'))
  if ref and ref in suno_by:
   pc.update(reference_audio='./'+suno_by[ref]['audio'].lstrip('./'),reference_label='真实Suno10s音色参考（歌词不同，非缺陷gold）')
  pairs.append(pc)
  if not aa.get('matched_comparator_id'):aa['matched_comparator_id']=bb['id']
  if not bb.get('matched_comparator_id'):bb['matched_comparator_id']=aa['id']
 for family in ['synth','acoustic']:
  for arm in ['literal_zh','literal_en','concrete']:
   pair(f'prompt-{family}-s37-{arm}',f'{family} · seed37 · 基线→{ARM_LABELS[arm]}',f'diff_{family}_s37_base',f'diff_{family}_s37_{arm}','同模型、同歌词、同初始噪声，比较实际文字条件作用；还需分别判断人工感、配器保留及质量。','提示变化会改变旋律/编曲，无法保证同一句对齐；提示分组不是人类缺陷标签。')
 for family in ['synth','acoustic']:
  pair(f'prompt-{family}-s91-concrete',f'{family} · seed91 · 基线→具体发声描述',f'diff_{family}_s91_base',f'diff_{family}_s91_concrete','第二个固定种子，检查观察是否只来自单个随机实现。','只有两种子和一套新歌词，不能据此推断总体提示遵循率。')
  pair(f'cfg-{family}-s37',f'{family} · native CFG0→2',f'diff_{family}_s37_cfg0',f'diff_{family}_s37_base','固定文字和噪声，仅比较guidance强度；native0已是条件生成。','公式为conditional+s(conditional−null)，null清空歌词与风格；不是负向提示实验。')
 for family in ['synth','acoustic']:
  da=f'diff_{family}_s37_audio_reference';ba=f'bloom_{family}_s37_audio_reference';ref=tracks.get(da,{}).get('reference_id')
  pair(f'model-{family}-audio-ref',f'共享{family}参考 · DiffRhythm2 / SongBloom',da,ba,'相同新歌词、同一真实Suno10s音频参考，比较两种真实模型生成；Suno味与不想要的伪影分开评分。','跨模型相同seed不代表同一初始噪声，生成旋律/配器仍不同；Suno源歌词也不同，模型来源不充当缺陷gold。',ref)
 # Muse variant IDs may differ after a checked parameter-route repair. Select by
 # actual used guidance and exact code hash rather than by requested argument.
 muse=[t for t in tracks.values() if t['id'].startswith('muse_') and case_clips[t['id']]]
 base_muse=[t for t in muse if t.get('arm')=='base' and t.get('actual_guidance_scale',t.get('guidance_scale')) is not None and not str(t.get('guidance_route','')).startswith('native_ignored')]
 def guidance(t):return float(t.get('actual_guidance_scale',t.get('guidance_scale')))
 if base_muse:
  low=min(base_muse,key=guidance);high=max(base_muse,key=guidance);mid=min(base_muse,key=lambda t:abs(guidance(t)-1.5))
  if low['id']!=high['id']:
   if low.get('exact_token_control_group')!=high.get('exact_token_control_group'):raise ValueError('Muse renderer token groups differ')
   if any(t.get('same_decoder_random_draws_as_first_identical_codes') is False for t in [low,high]):raise ValueError('Muse renderer noise mismatch')
   pair('muse-renderer-low-high','Muse固定token · renderer低/高guidance',low['id'],high['id'],'固定同一Muse音频token序列和decoder随机张量，只改变实际送入renderer的guidance。','已核对同token与同噪声，实际decoder guidance为1与2；只测这一组内容，尚无自然度结论。')
  concrete=[t for t in muse if t.get('arm')=='concrete' and abs(guidance(t)-1.5)<1e-5]
  if concrete:pair('muse-prompt-concrete','Muse真实提示变化 · 同renderer设置',mid['id'],concrete[0]['id'],'同一全局配器与新歌词，只改变Muse分段发声描述；renderer实际guidance均为1.5。','这对使用不同的真实Muse token，不能与固定token解码实验混同；HF采样音频预算有限。')
 # Quantitative changes are descriptive only, grouped by real generation seed.
 contrasts=[]
 for family in ['acoustic','synth']:
  for arm in ['literal_zh','literal_en','concrete']:
   observations=[]
   for seed in [37,91]:
    aa,bb=tracks.get(f'diff_{family}_s{seed}_base'),tracks.get(f'diff_{family}_s{seed}_{arm}')
    if not aa or not bb:continue
    observations.append(dict(seed=seed,base=aa['id'],changed=bb['id'],delta={k:(bb['window_mean'][k]-aa['window_mean'][k]) if finite(bb['window_mean'][k]) and finite(aa['window_mean'][k]) else None for k in METRICS},relative_change={k:relative(aa['window_mean'][k],bb['window_mean'][k]) for k in METRICS},base_values=aa['window_mean'],changed_values=bb['window_mean']))
   if observations:contrasts.append(dict(family=family,arm=arm,scope='Per-seed means of prescribed10s windows; not whole-song quality',observations=observations,negative_delta_counts={k:sum(finite(o['delta'][k]) and o['delta'][k]<0 for o in observations) for k in METRICS},seeds=len(observations),interpretation='No direction is a validated perceptual success label. No p-value or accuracy is computed.'))
 cfg=[]
 for family in ['acoustic','synth']:
  values=[]
  for level,cid in [(0,f'diff_{family}_s37_cfg0'),(1,f'diff_{family}_s37_cfg1'),(2,f'diff_{family}_s37_base')]:
   t=tracks.get(cid)
   if t:values.append(dict(native_cfg=level,standard_cfg_scale=level+1,case_id=cid,window_mean=t['window_mean']))
  if values:cfg.append(dict(family=family,seed=37,values=values))
 muse_checks=None
 muse_run_path=w/'muse_run.json'
 if muse_run_path.exists():
  mr=read(muse_run_path);native=[r for r in mr['samples'] if r.get('analysis_include') is False];actual=[r for r in mr['samples'] if r.get('analysis_include') is not False];hashes=[r['audio_sha256'] for r in native];pcm_hashes=[r.get('audio_pcm_sha256') for r in native]
  muse_checks=dict(native_api_audit_outputs=len(native),native_requested_scales=[r.get('guidance_scale') for r in native],native_effective_scales=[r.get('effective_guidance_scales') for r in native],native_container_hashes=hashes,native_pcm_hashes=pcm_hashes,native_different_arguments_identical_audio=(len(set(pcm_hashes))==1 if len(pcm_hashes)>=2 and all(pcm_hashes) else None),waveform_equality_method='Decodedfloat32PCM SHA256, not timestampedWAVcontainerSHA. Container hashes remain separately bound to scoring input.',analyzed_outputs=len(actual),exact_token_groups={r['id']:r.get('exact_token_control_group') for r in actual},actual_effective_scales={r['id']:r.get('effective_guidance_scales') for r in actual},same_decoder_random_draws_as_first_identical_codes={r['id']:r.get('same_decoder_random_draws_as_first_identical_codes') for r in actual},scope='Native argument-noop check versus actual per-instance inference_codes guidance override are separate experiments; no shared source files modified.')
 cross_model=[]
 for family in ['acoustic','synth']:
  for seed in [37,91]:
   da,bb=tracks.get(f'diff_{family}_s{seed}_audio_reference'),tracks.get(f'bloom_{family}_s{seed}_audio_reference')
   if da and bb:
    if da.get('reference_sha256')!=bb.get('reference_sha256'):raise ValueError('Cross-model reference mismatch')
    cross_model.append(dict(family=family,seed=seed,reference_id=da.get('reference_id'),reference_sha256=da.get('reference_sha256'),diff_case=da['id'],songbloom_case=bb['id'],diff_values=da['window_mean'],songbloom_values=bb['window_mean'],caveat='Same lyrics/reference, but different models, latent noise and generated music. Descriptor distance is not perceptual Suno likeness.'))
 conditioning=[]
 condition_file=w/'diff_conditioning.json'
 if condition_file.exists():
  embeddings=read(condition_file)['embeddings']
  for family in ['acoustic','synth']:
   base=tracks.get(f'diff_{family}_s37_base')
   if not base or base.get('prompt') not in embeddings:continue
   v=embeddings[base['prompt']]
   for arm in ['literal_zh','literal_en','concrete']:
    changed=tracks.get(f'diff_{family}_s37_{arm}')
    if not changed or changed.get('prompt') not in embeddings:continue
    z=embeddings[changed['prompt']];cos=sum(x*y for x,y in zip(v,z))/math.sqrt(sum(x*x for x in v)*sum(x*x for x in z))
    conditioning.append(dict(family=family,arm=arm,cosine_to_base=cos,encoder='MuQ-MuLan-large',caveat='Conditioning-vector diagnostic only; near cosine1 is not proof of instruction failure and is not audio quality.'))
 for c in contrasts:
  for o in c['observations']:
   aa,bb=tracks[o['base']],tracks[o['changed']];qa,qb=aa['audiobox_window_mean_PQ'],bb['audiobox_window_mean_PQ']
   o['audiobox_PQ_guard']=dict(base=qa,changed=qb,delta=(qb-qa) if finite(qa) and finite(qb) else None,note='Production-quality monitor, not artifact gold.')
 noise_checks=[]
 for family in ['acoustic','synth']:
  for seed in [37,91]:
   ts=[t for t in tracks.values() if t['id'].startswith('diff_') and t.get('family')==family and t.get('seed')==seed and t.get('arm')!='audio_reference'];nn=[t.get('initial_noise_sha256') for t in ts]
   if nn:noise_checks.append(dict(family=family,seed=seed,cases=len(ts),all_initial_noise_blocks_identical=all(x==nn[0] for x in nn),blocks_per_case=len(nn[0] or [])))
 if any(not x['all_initial_noise_blocks_identical'] for x in noise_checks):raise ValueError('Same-seed initial noise mismatch')
 aq_count=sum(r.get('audiobox') is not None for r in rows)
 findings=[f'已纳入{len(tracks)}个真实生成输出、{len(rows)}个固定10秒窗口；人工标签0，没有训练新的reward或RL。',f'DiffRhythm2每组外部设置种子并逐块核对初始噪声；{len(noise_checks)}组校验均一致。相同随机起点不保证乐句对齐。','中文笼统否定、英文笼统否定、具体发声描述和CFG扫描均保留实际输出；提示语本身不当成“成功减少伪影”的gold标签。','“Suno描述子距离”只测三个现有指标与这批Suno样本中心的距离，不能把它当Suno味、自然度或缺陷概率。','对比时必须同时听合成器是否保留、音质与音乐性。更软的编曲、较少鼓点、低音量或频宽变窄都可能让代理下降。']
 if muse_checks and muse_checks.get('native_different_arguments_identical_audio'):
  findings.insert(1,'已验证本实验归档MuCodec的原生guidance参数1与2生成的float32PCM逐位相同，实际模型调用仍为1.5；单改API参数不是有效guidance实验。另用实例级forward路由落实实际1/1.5/2，固定同一token和噪声。')
 if cfg and all(len(c['values'])==3 for c in cfg):
  description=[]
  for c in cfg:
   description.append(c['family']+' DW70 '+ '→'.join(number(v['window_mean']['roughness70'],3) for v in c['values']))
  findings.insert(2,'固定seed37，nativeCFG0→1→2的两个固定窗口均值：'+'；'.join(description)+'。粗糙度变化不等于人工感变化，需同时听配器与质量。')
 syn=next((c for c in contrasts if c['family']=='synth' and c['arm']=='concrete'),None)
 if syn and syn['seeds']==2 and syn['negative_delta_counts']['roughness70']==2 and syn['negative_delta_counts']['resonance_protected']==0:
  pq=[o['audiobox_PQ_guard']['delta'] for o in syn['observations']]
  msg='合成器条件下，具体发声描述使两个种子的DW70都下降，但保护共振都上升'
  if all(finite(x) and x<0 for x in pq):msg+='，AudioboxPQ也都下降（'+ '/'.join(number(x,3) for x in pq)+'）'
  findings.append(msg+'；不能将单一roughness下降当成功。')
 if aq_count:findings.append(f'Audiobox在{aq_count}/{len(rows)}个相同评分WAV上作为生产质量监测；不用于替代人工伪影标签。')
 priorities={'muse-renderer-low-high':0,'model-synth-audio-ref':1,'prompt-synth-s37-concrete':2}
 pairs.sort(key=lambda pair:priorities.get(pair['id'],3))
 resources=[]
 for group,m in manifest_files.items():
  ts=m['tracks'];gen_times=[t.get('end_to_end_seconds',t.get('inference_seconds',t.get('generation_seconds'))) for t in ts];gen_times=[x for x in gen_times if finite(x)]
  resources.append(dict(method='Muse MuCodec渲染（不含LM）' if group=='muse' else group+'真实生成',device='RTX5090；实际原生模型/解码器',timing=f"{len(ts)}个输出；每条中位{statistics.median(gen_times):.2f}s" if gen_times else '原始运行profile另存',notes='每条计时不含权重/模型加载；Muse只计MuCodec渲染，不含LM采样。同机已有任务保持运行，不是隔离吞吐benchmark。'))
 for group,pf in profiles.items():
  resources.append(dict(method=group+'心理声学评分',device='CUDA描述子＋CPU MoSQITo参考',timing=f"DW70 CPU整轮{pf['roughness'].get('total_seconds',0):.1f}s；{pf['roughness'].get('workers','?')}worker",notes='单一70dB RMS声压级分析假设；不是实际播放校准，也不是人类粗糙度/人工感标签。'))
 references=[dict(title='DiffRhythm2原生CFM与CFG实现',url='https://github.com/xiaomi-research/diffrhythm2/blob/main/diffrhythm2/cfm.py',code='https://github.com/xiaomi-research/diffrhythm2/blob/main/inference.py'),dict(title='SongBloom原论文：歌词与音频参考、full-ft使用Suno数据的说明',url='https://arxiv.org/abs/2506.07634'),dict(title='Muse原始论文：全局/分段风格控制',url='https://arxiv.org/abs/2601.03973',code='https://github.com/yuhui1038/Muse'),dict(title='Classifier-Free Diffusion Guidance',url='https://arxiv.org/abs/2207.12598'),dict(title='MoSQITo Daniel–Weber参考roughness',url='https://mosqito.readthedocs.io/en/latest/source/reference/mosqito.sq_metrics.roughness.roughness_dw.roughness_dw.html')]
 data=dict(schema_version=1,attribute_probe=True,suno_real=False,metadata=dict(version='prompt-control-20261105-v1',date='2026-11-05',human_labels_count=0,annotation_compatible_versions=[],previous_sample_ids=[],report_url=BASE+'PROMPT_CONTROL_REPORT.md',code_url=BASE+'PROMPT_CONTROL_REPORT.md#复现',limitations=['探索性同域控制实验：一套新歌词、两种风格、两个DiffRhythm2种子；没有总体提示遵循率结论。','同初始噪声/同token实验与跨模型参考比较必须分开解释。','所有主观栏初始待评价；模型来源、prompt分组和指标方向都不是缺陷gold。','Suno来源相似度与不想要的人工伪影分成两项听评，正常合成器不能直接扣分。','片段指标仅针对固定窗口；全曲MP3用于核对前后文，未把窗口均值冒充全曲音质。']),summary=dict(conclusion=('本实验归档MuCodec的外层guidance参数不生效已实证；真实prompt与有效guidance对照已补齐，仍需听评才能决定电音reward。' if muse_checks and muse_checks.get('native_different_arguments_identical_audio') else '真实prompt与guidance对照已生成并评分；目前用于核对控制能力和指标误伤，尚无可直接优化的电音reward。'),evidence=f'{len(tracks)}个真实生成输出、{len(rows)}个固定10秒窗口；人工听感标签0，新增RL训练0。',scope='先听同模型/种子对照，再看实际条件与指标；跨模型使用共同参考但不保证旋律或音色相同。'),samples=rows,tracks=list(tracks.values()),listening_pairs=pairs,findings=findings,methods=[dict(name='文字条件干预',measures='DiffRhythm2实际MuQ-MuLan条件；中文/英文否定与具体发声描述，保留合成器要求。',status='实际输出与固定窗口评分',limitations='否定理解与自然度需分别听评；同噪声不保证同一句对齐。'),dict(name='CFG强度/renderer参数核查',measures='nativeCFG0/1/2；Muse若纳入则记录实际guidance路由与固定token/噪声。',status='实际参数与哈希记录',limitations='CFG是采样执行器，不是伪影检测器；签名参数不保证实际生效。'),dict(name='三个心理声学描述子＋Suno描述子距离',measures='子带快速调制、DW70、带谐波保护的窄峰；相对小型Suno参考分布的三维距离。',status='相同评分WAV逐一SHA匹配',limitations='不是伪影或来源概率；正常鼓、和声、电子音色和内容差异会改变分数。')],resources=resources,recommendations=[dict(name='先定义听感，再冻结可验证的偏好目标',description='用真实Suno与真实控制输出记录人工伪影、Suno音色相似度、配器保留、音质与音乐性。少量精选标签只能用于探索，另取整首/整族新样本留出。',status='未训练人类偏好头，λ_art仍为0，仅shadow scoring。'),dict(name='CFG扫描与同tokenrenderer诊断优先于未验证reward guidance',description='先用实际可运行控制找到哪一层改变听感；若输入条件向量无法表达区别，负向条件外推也未必有用。',risk='引导可能放大伪影；直接−roughness或−共振可能损害正常合成器。')],references=references,figures=[dict(url='./visuals/prompt-control-muse-renderer.png',alt='固定Muse token和噪声的实际renderer guidance变化',caption='同一750音频token、相同decoder噪声；居中10秒。各指标独立纵轴，变化不等于听感改善。'),dict(url='./visuals/prompt-control-diff-cfg.png',alt='DiffRhythm2固定seed的CFG扫描',caption='相同歌词/style/初始噪声，seed37；两个固定10秒窗口均值。native参数0已经是条件生成。')],comparisons=[],checks=dict(human_labels_count=0,rl_training_runs=0,actual_generation_records=len(tracks),scored_clips=len(rows),audiobox_same_wav_scores=aq_count,input_status=status,input_sha256=input_sha,same_seed_noise_checks=noise_checks,muse_native_and_renderer_checks=muse_checks,reference_suno_audit_sha256=sha(a.suno_audit),selection_is_heldout_validation=False),model_provenance=model_provenance,analysis=dict(feature_distance_calibration=cal,prompt_contrasts=contrasts,cfg_sweep=cfg,cross_model_reference=cross_model,text_condition_diagnostics=conditioning,warning='All changes are descriptive. No human correlation, detection accuracy or RL improvement is inferred.'))
 # Fail closed on accidental serialization of internal paths or machine identity.
 encoded=json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
 for forbidden in ['/Users/','/home/','/storage/','/tmp/','token_file','reference_path']:
  if forbidden in encoded:raise ValueError('Private field leaked: '+forbidden)
 dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(encoded)
 analysis_path=dest.with_name('prompt-control-analysis.json');analysis_path.write_text(json.dumps(dict(checks=data['checks'],analysis=data['analysis'],tracks=data['tracks']),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 stream=io.StringIO();writer=csv.writer(stream,lineterminator='\n');writer.writerow(['id','model','family','arm','seed','start_s',*METRICS,'Audiobox_PQ','human_artificialness','human_quality','human_musicality','human_suno_likeness','human_instrument_retention'])
 for r in rows:writer.writerow([r['id'],r['model'],r['prompt_control']['family'],r['prompt_control']['arm'],r['prompt_control']['seed'],r['excerpt_start_s'],*[r['psycho'][k] for k in METRICS],r['audiobox'].get('PQ') if r.get('audiobox') else None,'','','','',''])
 dest.with_name('prompt-control-scores.csv').write_text(stream.getvalue());print(json.dumps(dict(samples=len(rows),tracks=len(tracks),pairs=len(pairs),groups=[x['group'] for x in status if x['status']=='included_complete_scores'],output=str(dest))))
 return data

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',required=True);p.add_argument('--suno-audit',required=True);p.add_argument('--groups',nargs='+',default=['diff','bloom','muse']);p.add_argument('--output',required=True);p.add_argument('--skip-missing',action='store_true');p.add_argument('--audiobox',default='audiobox_scores.jsonl');a=p.parse_args();build(a)
if __name__=='__main__':main()
