"""SHA-bound public allowlist for external two-conditional generation diagnostics."""
import argparse, hashlib, json, math, statistics, re
from pathlib import Path
import numpy as np
import soundfile as sf

BASE='https://github.com/zhanh-he/RL-paper-reading/blob/main/music-gen/reward-regime-study/naturalness/'
AUDIO='./audio/contrast-followup-20261105/'
METRICS=['modulation_depth','modulation_low','modulation_high','roughness70','resonance_protected']
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mean(xs): return statistics.mean(xs)
def regional(g, lo, hi):
    m=g['maps']; rows=[i for i,f in enumerate(m['center_hz']) if lo<=f<hi]
    weights=[(m['band_energy_relative'][i][j]**2 if m['band_energy_relative'][i][j]>.001 else 0) for i in rows for j in range(len(m['time_s']))]
    values=[m['modulation_depth'][i][j] for i in rows for j in range(len(m['time_s']))]
    return sum(x*w for x,w in zip(values,weights))/sum(weights)
def trace(g):
    m=g['maps']; values=[]
    for j in range(len(m['time_s'])):
        ws=[row[j]**2 if row[j]>.001 else 0 for row in m['band_energy_relative']]
        values.append(sum(row[j]*w for row,w in zip(m['modulation_depth'],ws))/sum(ws))
    return dict(time_s=m['time_s'],value=values)
def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',required=True);p.add_argument('--output',required=True);a=p.parse_args();w=Path(a.workspace);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    run=read(w/'run.json');manifest=read(w/'excerpts.json');au=read(w/'auditory.json');rr=read(w/'roughness.json');aq=[json.loads(line) for line in (w/'audiobox.jsonl').read_text().splitlines() if line.strip()]
    assert all(x['status']=='complete' for x in [run,au,rr]); assert len(run['samples'])==8
    assert run['profile']['batch_sha256']==sha(w/'contrast_batch.json')
    assert len(manifest['tracks'])==8 and len(manifest['samples'])==16
    assert len({r['id'] for r in manifest['samples']})==16
    assert run['profile']['shared_source_unchanged'] and len(run['profile']['sham_controls'])==2
    assert all(all(c[k] for k in ['same_latent','same_pcm','same_noise']) for c in run['profile']['sham_controls'])
    maps={r['id']:r for r in au['samples']};rough={r['id']:r for r in rr['samples']};quality={r['id']:r for r in aq};cases={r['id']:r for r in run['samples']}
    samples=[];raw=[];tracks=[]
    for r in manifest['samples']:
        rid=r['id'];g=maps[rid]['auditory'];roughrow=rough[rid];q=quality[rid];c=cases[r['source_case_id']]
        assert maps[rid]['audio_sha256']==roughrow['audio_sha256']==q['audio_sha256']==r['excerpt_sha256']
        assert r['source_original_sha256']==c['audio_sha256'] and r['source_run_sha256']==sha(w/'run.json')
        assert sha(w/'clips'/(rid+'.wav'))==r['excerpt_sha256'] and sha(w/'clips'/(rid+'.mp3'))==r['mp3_sha256']
        assert g['valid'] and roughrow['valid'] and q.get('scores') and c['lyric_prefix']['equal']
        wave,sr=sf.read(w/'clips'/(rid+'.wav'),dtype='float64',always_2d=True);assert sr==48000 and len(wave)==480000 and np.isfinite(wave).all()
        excerpt_signal=dict(rms_dbfs=float(20*np.log10(max(float(np.sqrt(np.mean(wave.mean(1)**2))),1e-12))),peak=float(np.abs(wave).max()))
        psy=dict(valid=True,modulation_depth=g['modulation_depth_20_300'],modulation_coherent=g['coherent_modulation_depth'],modulation_low=regional(g,0,1500),modulation_high=regional(g,3000,8000),roughness70=roughrow['asper']['mean'],roughness70_p95=roughrow['asper']['p95'],resonance_raw=g['isolated_resonance_proxy'],resonance_protected=g['harmonic_veto_resonance_proxy'],resonance_frequency_hz=g['harmonic_veto_candidate_hz'])
        sample=dict(id=rid,label=f"{c['family']} · {c['arm']} · β{c['beta']:g} · {r['excerpt_start_s']:g}–{r['excerpt_start_s']+10:g}s",kind='actual_generated_control',model='DiffRhythm2 · two-conditional external intervention',checkpoint='Frozen author weights; no RL',input_id=f"contrast/{c['family']}/s37",source_family=f"contrast/{c['family']}/ownlyrics-v1",pair_group=r['pair_group'],source_case_id=c['id'],source_original_sha256=r['source_original_sha256'],excerpt_sha256=r['excerpt_sha256'],mp3_sha256=r['mp3_sha256'],audio=AUDIO+rid+'.mp3',full_audio=AUDIO+c['id']+'__full.mp3',full_mp3_sha256=r['full_mp3_sha256'],duration_s=10.,sample_rate=48000,channels=r['channels'],excerpt_start_s=r['excerpt_start_s'],source_duration_s=r['source_duration_s'],window_index=r['window_index'],transform='none',selection_reason=r['selection_reason'],rights='Newly authored lyrics and prompts; public author-model generation, research demo. No training audio or weights redistributed.',human_rating=None,psycho=psy,prompt_control=dict(family=c['family'],arm=c['arm'],seed=c['seed'],contrast_beta=c['beta'],prompt='Positive: '+c['good_style']+' / Avoid: '+c['bad_style']),timeline=dict(modulation=trace(g),roughness=dict(time_s=roughrow['time_s'],value=roughrow['time_asper']),resonance=dict(time_s=g['resonance_time']['time_s'],value=g['resonance_time']['harmonic_veto_excess'])),audiobox=q['scores'],signal=excerpt_signal,source_url='https://github.com/xiaomi-research/diffrhythm2')
        samples.append(sample); raw.append(dict(id=rid,audio_sha256=r['excerpt_sha256'],auditory=g,roughness70={k:roughrow[k] for k in ['valid','asper','time_s','time_asper','specific_asper','bark_axis','seconds']},audiobox=q['scores']))
    for r in manifest['tracks']:
        cid=r['id'];c=cases[cid];ss=[s for s in samples if s['source_case_id']==cid]
        assert r['audio_sha256']==c['audio_sha256'] and sha(w/'clips'/(cid+'__full.mp3'))==r['full_mp3_sha256']
        assert [s['excerpt_start_s'] for s in ss]==[10.,40.]
        keep=['id','family','arm','seed','beta','sham','good_style','bad_style','sample_rate','channels','duration_s','audio_sha256','audio_pcm_sha256','latent_sha256','initial_noise_sha256','same_seed_noise_common_prefix_matches','complete_initial_noise_sequence','lyric_prefix','good_style_embedding_sha256','bad_style_embedding_sha256','good_bad_style_cosine','negative_style_calls','velocity_pair_calls','first_velocity','actual_sampler_kwargs','ode_seconds','decode_seconds','decoder_seed','end_to_end_seconds','gpu_peak_allocated_bytes','rms_dbfs','peak','human_rating']
        t={k:c[k] for k in keep};t.update(full_audio=AUDIO+cid+'__full.mp3',full_mp3_sha256=r['full_mp3_sha256'],window_mean={k:mean([s['psycho'][k] for s in ss]) for k in METRICS},audiobox_window_mean_PQ=mean([s['audiobox']['PQ'] for s in ss]),scored_windows=[dict(id=s['id'],start_s=s['excerpt_start_s'],end_s=s['excerpt_start_s']+10) for s in ss]);tracks.append(t)
    pairs=[]
    for family in ['synth','acoustic']:
        for arm in ['contrast05','contrast10','sham10']:
            aa=next(s for s in samples if s['source_case_id']==f'contrast_{family}_s37_conditional' and s['window_index']==1);bb=next(s for s in samples if s['source_case_id']==f'contrast_{family}_s37_{arm}' and s['window_index']==1)
            checkpoints=[]
            for s in [aa,bb]:
                for method in ['modulation','roughness','resonance']:
                    series=s['timeline'][method];i=max(range(len(series['value'])),key=lambda i:series['value'][i]);checkpoints.append(dict(sample_id=s['id'],clip_s=series['time_s'][i],source_s=s['excerpt_start_s']+series['time_s'][i],method=method))
            pairs.append(dict(id=f'contrast-{family}-{arm}',label=f'{family} · conditional / {arm}',a=aa['id'],b=bb['id'],purpose='检查两条件风格外推是否改变目标人工感，并独立判断音质、音乐性与配器保留。',observation='；'.join(f"{k}: {aa['psycho'][k]:.4f}/{bb['psycho'][k]:.4f}" for k in METRICS),caveat='同歌词、同初始噪声；条件变化仍会改变乐句/编曲。标签是干预意图，不是伪影gold。Sham使用完全相同条件，PCM已逐位核对。',checkpoints=checkpoints))
    profile=run['profile'];keep=['model','assets','asset_revision','native_source_sha256','sampler_patch','device','torch','requested_seconds','steps','odeint_kwargs','load_seconds','batch_sha256','inference','formula','branch_semantics','beta_zero','seed_protocol','scope','reproducibility','sham_controls','shared_source_unchanged']
    receipt=dict(schema='external-contrast-receipt-v1',status='complete',profile={k:profile[k] for k in keep},tracks=tracks)
    changes=[]
    for family in ['acoustic','synth']:
        low=next(t for t in tracks if t['family']==family and t['arm']=='conditional');high=next(t for t in tracks if t['family']==family and t['arm']=='contrast10')
        changes.append(f"{family} β0→1固定窗口均值：M {low['window_mean']['modulation_depth']:.4f}→{high['window_mean']['modulation_depth']:.4f}；DW70 {low['window_mean']['roughness70']:.4f}→{high['window_mean']['roughness70']:.4f}；PQ {low['audiobox_window_mean_PQ']:.4f}→{high['audiobox_window_mean_PQ']:.4f}。这是描述性变化，不能用来宣布伪影减少。")
    findings=changes+['本轮是两条保留相同歌词与配器的条件之间外推：v₊+β(v₊−v₋)；没有无条件分支，也没有学习artifact reward。','8次真实生成，含2次sham；两组sham与各自β0基线的latent、PCM和初始噪声逐位相同。','声学指标与PQ是描述性观测；没有人工标签，不能据此宣布减少电音感或保护了合成器。','只测一套歌词、两种风格、一个seed；固定初始噪声不保证生成乐句对齐。']
    audit=dict(schema_version=1,attribute_probe=True,contrast_followup=True,metadata=dict(version='contrast-followup-20261105-v1',date='2026-11-05',human_labels_count=0,report_url=BASE+'CONTRAST_FOLLOWUP_REPORT.md',limitations=['公共外部模型的探索，不推断其他renderer实现。','中心频率分组有重叠；不是校准响度或缺陷概率。']),summary=dict(conclusion='这轮验证条件对比引导是否可控；尚无听感证据支持作为电音reward或直接用于GRPO。',evidence=f"8次实际生成（含2次重复性检查）；{len(samples)}个固定10秒窗口；2/2 sham PCM与latent完全一致；人工标签0。",scope='β0/.5/1与原生CFG参数含义不同，不能混用。'),samples=samples,tracks=tracks,listening_pairs=pairs,findings=findings,methods=[],resources=[dict(title='完整外部实验与复现',url=BASE+'CONTRAST_FOLLOWUP_REPORT.md')],checks=dict(sham_controls=profile['sham_controls'],rl_training_runs=0,human_labels_count=0,scored_clips=len(samples),audiobox_same_wav_scores=len(quality),source_receipt_sha256=sha(w/'run.json')),timing=dict(generation_load_seconds=profile['load_seconds'],generation_seconds=sum(t['end_to_end_seconds'] for t in tracks),auditory=au['profile'],roughness=rr['profile']))
    for name,value in [('audit.json',audit),('receipt.json',receipt),('raw-scores.json',dict(samples=raw)),('audiobox-scores.json',dict(samples=aq))]:
        text=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
        assert not any(x in text for x in ['/Users/','/home/','/storage/']) and not re.search(r'\blab\d{4}\b',text)
        (out/name).write_text(text)
    lines=['# 外部条件对比引导：实测补充','', '2026-11-05。**工程控制已验证；听感效果待评价，不能用于证明artifact reward或GRPO改善。**','', '使用 DiffRhythm2 官方公开权重、同一套自写歌词、两种配器、seed37，真实生成8次60秒（含2次sham）。分支均保留相同歌词与生成历史，只改变风格条件。公式 `v_positive + β(v_positive − v_avoid)`，β=0/.5/1；这是两条件外推，区别于原生清空歌词和风格的CFG。提示写了自然/机器人不等于生成真的好/坏。','', '已核查：初始噪声序列一致；歌词前缀实际tensor一致；2/2 sham的latent和解码PCM逐位一致；共享源码未修改。固定10–20s与40–50s评分，原音量，无人工添加伪影。声压级统一70dB是DW分析假设；播放器MP3未做音量归一。','', '| 配器 | 条件 | 全带M | <1.5k中心组M | 3–8k中心组M | DW70 | 保护共振 | PQ监测 |','|---|---|---:|---:|---:|---:|---:|---:|']
    for t in tracks:
        vals=[t['window_mean'][k] for k in METRICS]+[t['audiobox_window_mean_PQ']];lines.append('| '+t['family']+' | '+t['arm']+' | '+' | '.join(f'{v:.4f}' for v in vals)+' |')
    lines += ['', '表中为两个固定窗口均值，不是整曲评估；没有已验证的“指标越低越自然”方向。分带M按听觉子带中心分组，频率响应重叠，并非互斥硬频带。','', '本轮修正先前acoustic具体提示中出现synth词汇的混杂：acoustic两分支均不提合成器，synth两分支均明确保留亮的模拟琶音、pad、电子bass与鼓机。不过措辞仍同时改变多项音色属性，只能解释为组合条件干预。只有一个seed；每个非sham干预未多次重复，也未验证更多歌词与内容，不能估计泛化收益。','', f"[试听与评分](https://zhanh-he.github.io/RL-paper-reading/demos/?nat-corpus=contrast&nat-pair=contrast-synth-contrast05#naturalness) · [代码](generate_diff_contrast.py) · [登记输入](runs/2026-11-05-contrast/batch.json) · [逐样本与有效参数](runs/2026-11-05-contrast/audit.json) · [实际调用证据](runs/2026-11-05-contrast/receipt.json)", '', '来源：[官方DiffRhythm2](https://github.com/xiaomi-research/diffrhythm2)；公开权重revision与SHA、实际耗时和显存见receipt。部署环境与缓存位置由运行者提供，源码/权重不会拷入本repo。','', '复现（先配置官方依赖与本地授权资产）：','```sh','python generate_diff_contrast.py --source "$DIFF_SOURCE" --assets "$DIFF_ASSETS" --runtime-root "$DIFF_RUNTIME" --batch runs/2026-11-05-contrast/batch.json --output "$RUN_OUTPUT" --seconds 60 --steps 32 --threads 4','python prepare_generation_excerpts.py --runs "$RUN_OUTPUT/run.json" --output-dir "$CLIPS" --manifest "$EXCERPTS"','python auditory_fulltrace.py --manifest "$EXCERPTS" --output "$AUDITORY" --device cuda --skip-roughness','OPENBLAS_NUM_THREADS=1 python score_roughness70.py --manifest "$EXCERPTS" --output "$ROUGHNESS" --workers 4','python score_learned.py --manifest "$EXCERPTS" --output "$QUALITY" --method audiobox --aes-checkpoint "$AUDIOBOX_WEIGHTS" --device cpu --threads 4','```','', '还需将本轮登记输入保存为 `contrast_batch.json`，并将评分原WAV、试听MP3及完整生成MP3置于该私有工作目录的 `clips/`（文件名保持case/window ID）。将同一轮run/excerpts/auditory/roughness JSON与audiobox JSONL命名为 `run.json`、`excerpts.json`、`auditory.json`、`roughness.json`、`audiobox.jsonl` 置于单个私有工作目录，再运行 `python build_contrast_followup.py --workspace "$PRIVATE_WORK" --output "$PUBLIC_JSON"`。原始WAV保留在repo之外；公开MP3另有编码指纹。','', f"实测生成加载{profile['load_seconds']:.1f}s；8次生成合计{sum(t['end_to_end_seconds'] for t in tracks):.1f}s；峰值本进程CUDA已分配{max(t['gpu_peak_allocated_bytes'] for t in tracks)/2**30:.2f}GiB（非整卡独占资源）；分析耗时见audit。两次sham不是额外独立音乐样本。"]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(tracks=len(tracks),samples=len(samples),pairs=len(pairs),sham='2/2 exact PCM+latent')))
if __name__=='__main__':main()
