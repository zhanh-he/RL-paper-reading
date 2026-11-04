"""Join scored real Suno excerpts, explicit metadata caveats, timelines and MP3s."""
import argparse,csv,hashlib,io,json,statistics
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];RUN=HERE/'runs/2026-11-04-suno-real'
URL='https://github.com/zhanh-he/RL-paper-reading/blob/main/music-gen/reward-regime-study/naturalness/'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',required=True);a=p.parse_args();w=Path(a.workspace);read=lambda f:json.loads((w/f).read_text())
 manifest=read('suno_manifest.json');features=read('suno_features.json');assert features['status']=='complete'
 executed=read('suno_executed_manifest.json');assert sha(w/'suno_executed_manifest.json')==features['profile']['manifest_sha256']
 assert {r['id']:r['excerpt_sha256'] for r in executed['samples']}=={r['id']:r['excerpt_sha256'] for r in manifest['samples']}
 pubs={r['id']:r for r in read('suno_public.json')['samples']};scores={r['id']:r for r in features['samples']};signals={r['id']:r for r in read('suno_signal.json')['rows']}
 assert len(scores)==len(pubs)==len(manifest['samples'])==48
 rows=[]
 for m in manifest['samples']:
  rid=m['id'];f=scores[rid];g=f['auditory'];rr=f['roughness'];q=rr['levels']['70']['asper']
  assert sha(m['path'])==m['excerpt_sha256']==f['audio_sha256']==signals[rid]['audio_sha256']==pubs[rid]['excerpt_sha256'];assert g['valid'] and rr['valid']
  maps=g['maps'];depth=np.asarray(maps['modulation_depth']);energy=np.asarray(maps['band_energy_relative']);weights=energy**2*(energy>.001);mod=(depth*weights).sum(0)/np.maximum(weights.sum(0),1e-20)
  timeline=dict(modulation=dict(time_s=maps['time_s'],value=mod.tolist()),roughness=dict(time_s=rr['levels']['70']['time_s'],value=rr['levels']['70']['time_asper']),resonance=dict(time_s=g['resonance_time']['time_s'],value=g['resonance_time']['harmonic_veto_excess']))
  psycho=dict(valid=True,modulation_depth=g['modulation_depth_20_300'],modulation_coherent=g['coherent_modulation_depth'],modulation_peak_hz=g['fast_modulation_peak_hz'],roughness70=q['mean'],roughness70_p95=q['p95'],roughness60=rr['levels']['60']['asper']['mean'],roughness80=rr['levels']['80']['asper']['mean'],roughness_raw=rr['raw_assumed']['asper']['mean'],resonance_raw=g['isolated_resonance_proxy'],resonance_3_8=g['resonance_3_8_proxy'],resonance_protected=g['harmonic_veto_resonance_proxy'],resonance_frequency_hz=g['harmonic_veto_candidate_hz'])
  allowed=['source_family','same_input_verified','publisher_style','song_id','track_index','excerpt_start_s','source_fraction','source_duration_s','source_codec','source_revision','metadata_lyrics_sha256','metadata_style_sha256']
  rows.append(pubs[rid]|{k:m[k] for k in allowed}|dict(psycho=psycho,signal=signals[rid]['signal'],timeline=timeline,human_rating=None))
 keys=['modulation_depth','modulation_coherent','roughness70','roughness70_p95','resonance_raw','resonance_protected'];distributions={k:dict(minimum=min(r['psycho'][k] for r in rows),median=statistics.median(r['psycho'][k] for r in rows),maximum=max(r['psycho'][k] for r in rows)) for k in keys}
 correlation={f'{a}:{b}':float(spearmanr([r['psycho'][a] for r in rows],[r['psycho'][b] for r in rows]).statistic) for a,b in [('modulation_depth','roughness70'),('modulation_depth','resonance_protected'),('roughness70','resonance_protected')]}
 level_ranks={str(level):float(spearmanr([r['psycho']['roughness70'] for r in rows],[r['psycho'][f'roughness{level}'] for r in rows]).statistic) for level in [60,80]}
 families=sorted({r['source_family'] for r in rows});pairs=[]
 # Fixed family/position order, with the electronic-labelled family first.
 order=['suno_cn_008001']+[f for f in families if f!='suno_cn_008001']
 for family in order:
  for fraction in [.5,.2,.8]:
   members=[r for r in rows if r['source_family']==family and r['source_fraction']==fraction];assert len(members)==2
   pairs.append(dict(label=f'{family} · 两个原始track · 相对位置{fraction:.0%}（非同句对齐）',a=members[0]['id'],b=members[1]['id']))
 by={r['id']:r for r in rows}
 for r in rows:r['matched_comparator_id']=next(x['id'] for x in rows if x['pair_group']==r['pair_group'] and x['id']!=r['id'])
 prof=features['profile'];times=[r['timing']['auditory_seconds'] for r in scores.values()];cpu=[r['timing']['roughness_cpu_seconds'] for r in scores.values()]
 checks=dict(real_suno_songs=16,source_families=8,clips=48,transformations=0,human_labels_count=0,model_training_runs=0,same_input_verified_pairs=0,pair_metadata=manifest['metadata_pair_checks'],feature_distributions=distributions,metric_spearman=correlation,cross_spl_rank_spearman=level_ranks,resonance_candidate_frequency_range_hz=[min(r['psycho']['resonance_frequency_hz'] for r in rows),max(r['psycho']['resonance_frequency_hz'] for r in rows)],resonance_candidates_at_least3khz=sum(r['psycho']['resonance_frequency_hz']>=3000 for r in rows),median_harmonic_veto_relative_reduction=float(np.median([1-r['psycho']['resonance_protected']/r['psycho']['resonance_raw'] for r in rows])),correlation_note='Metric-to-metric or assumed-level stability only, not human agreement;48 clips are clustered within16 songs/8 families.',source_dataset_revision=manifest['source_dataset_revision'],feature_script_sha256=sha(HERE/'psychoacoustic_features.py'),execution_manifest_sha256=sha(w/'suno_executed_manifest.json'),curated_metadata_manifest_sha256=sha(w/'suno_manifest.json'))
 auxiliary=json.loads((HERE/'runs/2026-11-04-psychoacoustic/audit.json').read_text())
 methods=[m|dict(status='48条真实Suno片段实测') for m in auxiliary['methods']]
 summary=dict(conclusion='真实Suno的三类特征已跑通；目前推荐作局部听评定位和后续偏好头输入，尚不能直接用于GRPO压制“电音感”。',evidence='Muse公开SunoV5训练音频：16首、8个同族ID、48条未加工10秒片段。5090运行子带/共振，CPU运行参考roughness；人工缺陷标签0，RL训练0。',scope='默认展示真实Suno；旧47条模拟/代理探针在辅助入口。歌曲来源不是缺陷标签，正常合成器不预判为坏。')
 findings=[f"三项在真实Suno中有可测差异：调制深度{distributions['modulation_depth']['minimum']:.3f}–{distributions['modulation_depth']['maximum']:.3f}；DW70 {distributions['roughness70']['minimum']:.3f}–{distributions['roughness70']['maximum']:.3f} asper；保护共振{distributions['resonance_protected']['minimum']:.4f}–{distributions['resonance_protected']['maximum']:.4f}。这些范围不是缺陷严重度。",
 f"调制与roughness的片段Spearman为{correlation['modulation_depth:roughness70']:.3f}；roughness与保护共振为{correlation['roughness70:resonance_protected']:.3f}。说明三项不等价，不证明哪项更符合听感。",
 '电子/合成器风格标注的cn008001是重点误伤试听组；标注来自发布者，不代表听者已经认为这些音色正常或有缺陷。其余样本多为流行/抒情/民谣/摇滚，电子音色覆盖仍窄。',
 '8个同族ID的两个track均未核实为完全相同输入。仅2组的归一化发布歌词完全相同，1组的发布style完全相同；这些是输出元数据，不能冒充原始API输入。',
 f"保护共振候选频率全落在{min(r['psycho']['resonance_frequency_hz'] for r in rows):.0f}–{max(r['psycho']['resonance_frequency_hz'] for r in rows):.0f}Hz，未落在3–8kHz；这暴露了固定窄峰代理可能偏向低中频残余谱峰，不能声称已抓到目标高频人工感，也不能据此说歌曲没有伪影。",
 '没有人类缺陷标签：无法报告检出率、误报率、人类排序一致性，也没有证明能压制真实Suno人工感。曲线峰值只用于定位试听，可能是鼓点、和声或正常电子音色。']
 data=dict(schema_version=1,suno_real=True,metadata=dict(version='suno-real-20261104-v1',date='2026-11-04',human_labels_count=0,report_url=URL+'SUNO_REAL_REPORT.md',code_url=URL+'SUNO_REAL_REPORT.md#复现',background_document={'status':'read'},limitations=['固定分片首两个成员，不是代表总体的随机样本；未按听感挑选。','相对时间位置未按乐句对齐；输出编曲/内容差异会干扰偏好。','源是有损MP3；评分用解码float WAV，在线试听为二次MP3编码，哈希各自记录。','70dB为分析声压级假设，不是实际播放校准或等sone。','尚无人工标签，无法决定哪个特征真正覆盖不想要的人工伪影。'],annotation_compatible_versions=[],previous_sample_ids=[]),summary=summary,samples=rows,findings=findings,listening_pairs=pairs,checks=checks,profile=prof,methods=methods,comparisons=[],resources=[dict(method='子带调制＋孤立共振',device='RTX5090 / CUDA / batch1',timing=f'10秒片段中位{statistics.median(times):.4f}s；PyTorch峰值分配{prof["gpu_peak_allocated_bytes"]/1e9:.2f}GB',notes='同步计时；不含解码/初始化。未扰动已有GPU任务。'),dict(method='Daniel–Weber参考roughness',device='CPU4worker；每worker BLAS1线程；60/70/80dB＋固定FS→Pa',timing=f'每片段四电平CPU中位{statistics.median(cpu):.1f}s；48条整轮wall{prof["total_seconds"]:.1f}s',notes='整轮wall含GPU/解码/序列化/CPU等待，不是CUDA纯耗时。参考roughness先离线用，不直接放rollout热路径。')],recommendations=[dict(name='先做真实Suno的缺陷偏好头（未训练）',description='分别判断人工感、音质、音乐性，给出伪影类型及正常电子音色保护标签。以调制、共振、roughness及必要音频embedding作为候选输入；以同族整歌分组留出，而非随机切片。',formula='候选：R_art = calibrate(hθ(robust_scale(f(x))))；自然度偏好更高→reward更高。不是固定−roughness，也不是DDSP参考重建loss。',integration='R_total = R_existing + λ_art·R_art − λ_guard·P_invalid。尚无听评校准时λ_art=0，仅shadow scoring；通过独立同域偏好与音色保护验收后再选权重。',status='没有训练/校准权重；不能声称已可改善GRPO。',risk='低通、降音量、简化配器、减弱鼓点或取消正常合成器都可能取巧。必须同时看音质/音乐性并审计目标生成模型的新rollout。'),dict(name='roughness先当离线解释指标',description='参考模型慢且粗糙感可以是合法音乐音色。GPU描述子速度可用于后续小头，但不是已验证reward。',status='本轮实测，未通过听感有效性验收')],references=[dict(title='Muse官方数据卡：SunoV5与MIT声明',url='https://huggingface.co/datasets/bolshyC/Muse/blob/'+manifest['source_dataset_revision']+'/README.md',code='https://github.com/yuhui1038/Muse'),dict(title='Muse原始论文',url='https://arxiv.org/abs/2601.03973'),*auxiliary.get('references',[])],figures=[dict(url='./visuals/suno-real-distributions.png',alt='48条真实Suno片段的三类特征与电平分布',caption='固定取样的真实输出；每个点是10秒片段，不是缺陷标签。48点来自16首歌/8个同族，不是48个独立样本。'),dict(url='./visuals/suno-real-family-p50.png',alt='相对位置50%的真实Suno同族两track比较',caption='连线只表示同族ID，未验证为完全相同输入，也非同一乐句。分数高不意味着更不自然。')])
 RUN.mkdir(parents=True,exist_ok=True);(RUN/'audit.json').write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n');(RUN/'features.json').write_text(json.dumps(features,allow_nan=False)+'\n');(RUN/'checks.json').write_text(json.dumps(checks,indent=2,allow_nan=False)+'\n')
 stream=io.StringIO();writer=csv.writer(stream,lineterminator='\n');writer.writerow(['id','source_family','song_id','start_s','publisher_style',*keys,'human_artificialness','human_quality','human_musicality'])
 for r in rows:writer.writerow([r['id'],r['source_family'],r['song_id'],r['excerpt_start_s'],r['publisher_style'],*[r['psycho'][k] for k in keys],'','',''])
 (RUN/'sample-scores.csv').write_text(stream.getvalue());print(json.dumps(dict(checks=checks,profile=prof),ensure_ascii=False))
if __name__=='__main__':main()
