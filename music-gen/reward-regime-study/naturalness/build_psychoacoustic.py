"""Publish only observed, SHA-bound psychoacoustic data and explicit probes."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics
import numpy as np
from scipy.stats import spearmanr

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];RUN=HERE/'runs/2026-11-04-psychoacoustic'
BASE='https://github.com/zhanh-he/RL-paper-reading/blob/main/music-gen/reward-regime-study/naturalness/'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',required=True);a=p.parse_args();w=Path(a.workspace)
    read=lambda f:json.loads((w/f).read_text())
    rough=read('psycho_roughness_results.json');aud=read('psycho_auditory_final.json');manifest=read('psycho_manifest.json')['samples']
    assert rough['status']==aud['status']=='complete';assert len(rough['samples'])==len(aud['samples'])==len(manifest)==47
    old=json.loads((HERE/'runs/2026-11-04/audit.json').read_text());pub={r['id']:r for r in old['samples']}
    pub.update({r['id']:r for r in read('psycho_public_new.json')['samples']})
    rs={r['id']:r for r in rough['samples']};gs={r['id']:r for r in aud['samples']};ss={r['id']:r for r in read('psycho_signal_results.json')['rows']}
    rows=[];full=[]
    for r in manifest:
        rid=r['id'];actual=sha(r['path']);g=gs[rid]['auditory'];rr=rs[rid]['roughness']
        assert actual==r['excerpt_sha256']==rs[rid]['audio_sha256']==gs[rid]['audio_sha256']==ss[rid]['audio_sha256']==pub[rid]['excerpt_sha256'],rid
        levels=rr['levels'];q70=levels.get('70',{}).get('asper',{})
        valid=g['valid'] and rr['valid']
        metrics=dict(valid=valid,modulation_depth=g['modulation_depth_20_300'] if valid else None,
                     modulation_coherent=g.get('coherent_modulation_depth') if valid else None,
                     modulation_peak_hz=g.get('fast_modulation_peak_hz') if valid else None,
                     roughness70=q70.get('mean'),roughness70_p95=q70.get('p95'),
                     roughness60=levels.get('60',{}).get('asper',{}).get('mean'),roughness80=levels.get('80',{}).get('asper',{}).get('mean'),
                     roughness_raw=rr.get('raw_assumed',{}).get('asper',{}).get('mean') if rr.get('raw_assumed') else None,
                     resonance_raw=g['isolated_resonance_proxy'] if valid else None,
                     resonance_3_8=g.get('resonance_3_8_proxy') if valid else None,
                     resonance_protected=g.get('harmonic_veto_resonance_proxy') if valid else None,
                     resonance_frequency_hz=g.get('harmonic_veto_candidate_hz'))
        extra={k:r[k] for k in ['common_headroom_gain','protected_style_control','comparison_id'] if k in r}
        row=pub[rid]|extra|dict(psycho=metrics,signal=ss[rid]['signal'],human_rating=None)
        if rid.startswith('psy_'):row['selection_reason']=r['selection_reason'];row['matched_comparator_id']=r.get('comparison_id')
        rows.append(row);full.append(dict(id=rid,audio_sha256=actual,auditory=g,roughness=rr,timing=dict(auditory_seconds=gs[rid]['timing']['auditory_seconds'],roughness_seconds=rs[rid]['timing']['roughness_cpu_seconds'])))
    by={r['id']:r for r in rows};comparisons=[];pairs=[]
    def pair(label,aa,bb,interpretation):
        A,B=by[aa]['psycho'],by[bb]['psycho']
        observation=f"调制深度 {A['modulation_depth']:.3f}→{B['modulation_depth']:.3f}；DW70 {A['roughness70']:.3f}→{B['roughness70']:.3f} asper；谐波保护共振 {A['resonance_protected']:.4f}→{B['resonance_protected']:.4f}"
        comparisons.append(dict(label=label,a=aa,b=bb,observation=observation,interpretation=interpretation));pairs.append(dict(label=label,a=aa,b=bb))
    for name in ['muse_baseline','ref_vibeace']:
        prefix='psy_'+name+'__'
        pair(name+'：原片→深70Hz调制',prefix+'anchor',prefix+'am70_deep','两条同源组均随调制强度增加；这验证对操作的响应，不等于人工缺陷排序。')
        pair(name+'：弱→强4.2kHz窄带音',prefix+'tone4200_weak',prefix+'tone4200_strong','保护后的共振分数随强度增加；roughness可能下降，不能只靠roughness。')
        pair(name+'：原片→突发共振',prefix+'anchor',prefix+'ring4200_bursts','已知增加的共振；不证明与真实Suno伪影相同，需听者判断。')
        pair(name+'：原片→明亮EQ音色对照',prefix+'anchor',prefix+'high_shelf_style','意图为音色保护对照；不能把任何特征增加都视为缺陷。')
        pair(name+'：原片→梳状滤波音色对照',prefix+'anchor',prefix+'comb_style','效果可被音乐有意使用；不自动贴坏标签。')
    pair('普通→明亮合成器','control_normal_synth','control_bright_synth','DW70约增加16倍；谐波保护抑制窄峰分数。两音色均未获人工自然度标签。')
    pair('明亮合成器→FM铃声音色','control_bright_synth','psy_control_fm_bell_style','FM铃声仍被共振代理打高分：谐波保护不覆盖所有正常电子音色。')
    pair('70Hz AM分析锚点→锯齿低音音色','psy_control_am1k_70hz','psy_control_saw_bass_style','锯齿音色的roughness更高；不能把粗糙感直接解释成不想要的伪影。')
    pair('Muse：原音量→低20dB','muse_baseline','muse_baseline__gain_minus20db','归一到同RMS声压级的结果近似不变；原电平假设的roughness会变，需防降音量捷径。')
    pair('VibeAce：原片→RMS匹配硬削波','ref_vibeace','ref_vibeace__distorted','共振代理大幅降低，说明失真可以逃过该单项；需要缺陷类型分头。')
    gpu=[r['timing']['auditory_seconds'] for r in full];cpu=[r['timing']['roughness_seconds'] for r in full if r['roughness']['valid']]
    std=[r['roughness']['levels']['70']['asper']['mean'] for r in full if r['roughness']['valid']]
    ranks={str(s):float(spearmanr(std,[r['roughness']['levels'][str(s)]['asper']['mean'] for r in full if r['roughness']['valid']]).statistic) for s in [60,80]}
    checks=dict(silent_invalid=not by['control_silence']['psycho']['valid'],human_labels_count=0,
                cross_level_rank_spearman=ranks,
                level_note='Cross-level metric stability only; not correlation with humans',
                anchor60_asper=rs['psy_control_am1k_70hz']['roughness']['levels']['60']['asper']['mean'],
                gain_invariance={name:{key:abs(by[name]['psycho'][key]-by[name+'__gain_minus20db']['psycho'][key]) for key in ['modulation_depth','roughness70','resonance_protected']} for name in ['muse_baseline','ref_vibeace']})
    profile=dict(auditory_gpu=aud['profile'],roughness_cpu_run=rough['profile'],auditory_median_seconds=statistics.median(gpu),roughness_median_seconds=statistics.median(cpu),
                 roughness_replacement=rough.get('replacement_profile'),roughness_replacement_note=rough.get('replacement_note'),
                 maximum_cpu_worker_rss_bytes=max(r['roughness'].get('worker_peak_rss_bytes',0) for r in full),
                 auditory_script_sha256=sha(HERE/'psychoacoustic_features.py'),roughness_executed_script_sha256=sha(w/'psycho_roughness_executed.py'))
    data=dict(schema_version=1,metadata=dict(version='psychoacoustic-20261104-v1',date='2026-11-04',human_labels_count=0,
          annotation_compatible_versions=['naturalness-20261104-v1'],previous_sample_ids=[r['id'] for r in old['samples']],
          report_url=BASE+'PSYCHOACOUSTIC_REPORT.md',code_url=BASE+'PSYCHOACOUSTIC_REPORT.md#复现',
          limitations=['47条为辅助机制/误伤探针；真实Suno主实验另列，不能以这些探针证明Suno缺陷识别有效。','70dB SPL是同RMS声压级假设，不是等sone或真实用户播放电平。','原25条音频未变，既有人工评分可兼容导入；新增22条没有虚构MOS。',
                       'GPU特征为自定义描述子；MoSQITo Daniel–Weber roughness在CPU计算，未称自定义代理为ECMA标准。']),
          summary=dict(conclusion='辅助探针：三类特征测量不同结构，并有正常电子音色误伤。此结果不替代真实Suno主实验，也不是可直接用于RL的reward。',
             evidence='5090实跑47条10s音频；包含原25条＋22条新探针与正常音色对照。参考roughness覆盖60/70/80dB及原电平假设；人工标签0，无新RL训练。',
             scope='原25条沿用已有AI分数；新增22条只做心理声学与信号分析，未补造AI分。'),samples=rows,
          findings=['70Hz调制从无→轻→深，两组音频的调制深度与DW roughness均递增。',
                    '增加4.2kHz窄带音时，谐波保护共振分数上升，roughness反而可能下降：指标测的是不同听感。',
                    '明亮合成器的原始窄峰分数0.599→谐波保护0.0002；FM铃声仍为0.393，误伤风险没有消失。',
                    f"谐波截止16kHz的锯齿低音DW70为{by['psy_control_saw_bass_style']['psycho']['roughness70']:.3f} asper，高于两组深AM探针；直接惩罚roughness会损害合法电子音色。"],
          listening_pairs=pairs,comparisons=comparisons,checks=checks,profile=profile,
          methods=[dict(name='听觉子带调制（自定义CUDA）',measures='32个ERB间隔四阶gammatone FIR；150Hz–16kHz，Hilbert包络→低通400Hz→1kHz；500ms窗/250ms hop，20–300Hz调制深度。附调制率集中度。',status='47条实测',limitations='不是asper/ECMA roughness；没有响度压缩和完整掩蔽模型。瞬态、和声拍频也会升高。'),
                   dict(name='Daniel–Weber roughness / MoSQITo1.2.1',url='https://mosqito.readthedocs.io/en/latest/source/reference/mosqito.sq_metrics.roughness.roughness_dw.roughness_dw.html',measures='200ms窗，50%重叠，47个Bark通道的参考模型；asper；三种同RMS声压级和一种固定FS→Pa假设。',status='47条实测，静音拒绝',limitations='roughness本身不等于音乐缺陷。CPU参考较慢，原电平假设受音量影响。'),
                   dict(name='孤立共振＋谐波保护（自定义CUDA）',measures='250Hz–16kHz局部窄峰/持久性；另报3–8kHz。尝试用强低频谱峰与高次谱峰解释谐波，保护普通合成音色。',status='47条实测',limitations='非严格可听性判据。纯音/FM铃声仍高；谐波排除也可能漏掉伪影。不能称伪影概率。')],
          resources=[dict(method='本轮听觉子带/共振',device='RTX5090 / CUDA / batch1',timing=f"单条10s中位 {statistics.median(gpu):.4f}s；47条总处理 {aud['profile']['total_seconds']:.2f}s；PyTorch峰值分配 {aud['profile']['gpu_peak_allocated_bytes']/1e9:.2f}GB",notes='总处理含解码/序列化，不含Python导入；GPU并非整个主机显存。'),
                     dict(method='本轮DW roughness',device='同机CPU /4 worker，每worker BLAS1线程',timing=f"每条四种电平合计中位 {statistics.median(cpu):.1f}s；初轮wall {rough['profile']['total_seconds']:.1f}s＋带限控制复算 {rough.get('replacement_profile',{}).get('total_seconds',0):.1f}s",notes='最终47条包括184次完整10s参考计算；被替换的旧锯齿结果未进入最终集。CPU与GPU模型范围不同，不作速度排行。未使用排队集群。')],
          figures=[dict(url='./visuals/psychoacoustic-response.png',alt='心理声学处理强度响应及正常音色误伤风险',caption='已知处理的响应和 intended style controls；没有人类缺陷gold，不能计算检出准确率。'),
                   dict(url='./visuals/psychoacoustic-time.png',alt='同源AM对照的听觉子带调制时频图与roughness曲线',caption='Muse同RMS对照。色标一致，图中变化是描述子响应；请结合播放器判断是否像目标伪影。')])
    RUN.mkdir(parents=True,exist_ok=True)
    (RUN/'audit.json').write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (RUN/'features.json').write_text(json.dumps(dict(profile=profile,samples=full),allow_nan=False)+'\n')
    (RUN/'checks.json').write_text(json.dumps(checks,indent=2,allow_nan=False)+'\n')
    stream=io.StringIO();writer=csv.writer(stream,lineterminator='\n');keys=['modulation_depth','modulation_coherent','roughness70','roughness70_p95','roughness60','roughness80','roughness_raw','resonance_raw','resonance_protected']
    writer.writerow(['id','kind',*keys,'human_artificialness','human_quality','human_musicality'])
    for r in rows:writer.writerow([r['id'],r['kind'],*[r['psycho'][k] for k in keys],'','',''])
    (RUN/'sample-scores.csv').write_text(stream.getvalue())
    print(json.dumps(dict(samples=len(rows),checks=checks,profile=profile),ensure_ascii=False))


if __name__=='__main__':main()
