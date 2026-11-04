"""Join hash-bound observed results and publish a deliberately sanitized snapshot."""
import argparse
import csv
import io
import json
from pathlib import Path
import shutil
import statistics

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
RUN=HERE/'runs/2026-11-04'
REPO='https://github.com/zhanh-he/RL-paper-reading/blob/main/music-gen/reward-regime-study/naturalness/'


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',required=True);args=p.parse_args()
    work=Path(args.workspace);RUN.mkdir(parents=True,exist_ok=True)
    read=lambda f:json.loads((work/f).read_text())
    manifest=read('public_manifest.json');learned=read('learned_results.json');signals=read('signal_results.json');song=read('songeval_results.json')
    li={r['id']:r for r in learned['samples']};si={r['id']:r for r in signals['rows']};so={r['id']:r for r in song['samples']}
    samples=[]
    for public in manifest['samples']:
        rid=public['id'];a=li[rid];s=si[rid];e=so[rid]
        assert public['excerpt_sha256']==a['audio_sha256']==s['audio_sha256']==e['audio_sha256'],rid
        samples.append(public|dict(signal=s['signal'],audiobox=a['audiobox'],
                                   pam=a['pam']|dict(score=a['pam']['PAM'],artifact_score=a['pam']['artifact_prompt_probe']),
                                   songeval=e['scores']|dict(mean=statistics.mean(e['scores'].values())),
                                   audiobox_segments=a['audiobox_segments'],songeval_segments=e['segment_spans'],
                                   timing=dict(audiobox_seconds=a['audiobox_seconds'],pam_seconds=a['pam_seconds'],
                                               signal_seconds=s['elapsed_s'],songeval_seconds=e['inference_seconds'])))
    byid={r['id']:r for r in samples}
    comparisons=[]
    for d in learned['paired_deltas']:
        # AM is interpreted only against its equal-RMS pure-gain comparator.
        if d['id'].endswith('__metallic_mod40hz') and not d['comparator_id'].endswith('__mod40hz_gain_control'): continue
        a,b=byid[d['id']],byid[d['comparator_id']]
        music=a['songeval']['Musicality']-b['songeval']['Musicality']
        comparisons.append(dict(id=d['id'],comparator_id=d['comparator_id'],
                                label=a['label']+' vs '+b['label'],
                                observation=f"PQ Δ {d['audiobox_PQ_delta']:+.3f}；PAM Δ {d['PAM_delta']:+.3f}；SongEval Musicality Δ {music:+.3f}",
                                interpretation='同源已知变换的自动响应；不是人工缺陷标签。AM只与等RMS纯增益对照比较。',
                                pq_delta=d['audiobox_PQ_delta'],pam_delta=d['PAM_delta'],musicality_delta=music))
    silence=byid['control_silence'];normal=byid['control_normal_synth'];bright=byid['control_bright_synth']
    comparisons.extend([
        dict(label='静音',observation=f"PQ {silence['audiobox']['PQ']:.3f}；PAM {silence['pam']['score']:.3f}；自拟prompt {silence['pam']['artifact_score']:.3f}",interpretation='评分器仍输出数值；整条静音应由任务有效性检查拒绝，不能用高PQ代表自然。'),
        dict(label='正常谐波合成器 → 明亮版本',observation=f"PQ {normal['audiobox']['PQ']:.3f}→{bright['audiobox']['PQ']:.3f}；PAM {normal['pam']['score']:.3f}→{bright['pam']['score']:.3f}",interpretation='两条是设计为合法音色的控制信号；没有人类标签，不能宣布误判率。但PAM对明亮度的响应值得听测。'),
        dict(label='自拟自然/伪影prompt对比',observation=f"{sum(r['pam']['artifact_score']>.99 for r in samples)}/25 分数 >0.99",interpretation='明显饱和且静音也高；本轮淘汰为直接reward，保留失败诊断。')])
    resources=[]
    for name,meta in learned['methods'].items():
        resources.append(dict(method=name,device=meta['hardware']+' / CPU4线程',
                              timing=f"10s片段中位 {meta['timing_summary']['median_seconds_per_10s_clip']:.3f}s；初始化 {meta['model_init_seconds']:.2f}s；进程峰值RSS {meta['peak_process_rss_bytes']/1e9:.2f}GB",
                              notes=meta['timing_scope']))
    sm=song['method']
    resources.append(dict(method='SongEval',device=sm['hardware']+' / CPU4线程',
                          timing=f"10s片段中位 {sm['median_seconds_per_10s_clip']:.3f}s；初始化 {sm['model_init_seconds']:.2f}s；峰值RSS {sm['peak_process_rss_bytes']/1e9:.2f}GB",
                          notes='复用现有Linux缓存，未使用GPU。'+sm['timing_scope']))
    resources.append(dict(method='Signal diagnostics',device='Apple M4 CPU / 单进程',timing=f"10s片段中位 {statistics.median(r['elapsed_s'] for r in signals['rows']):.3f}s（含解码）",notes='NumPy/SciPy；仅描述性特征，无自然度总分。'))
    candidates=json.loads((HERE/'candidates.json').read_text())['candidates']
    methods=[dict(name=c['name'],url=c.get('code'),measures=c['output']+'；'+c['domain'],
                  status='本轮已运行25条' if c['id'] in ['audiobox_pq_guardrail','pam_exploratory','songeval'] else '接口已实现，未训练' if c['id']=='domain_preference_head' else '一手核实，未在本轮运行',
                  limitations=c['risk']+'；weights: '+str(c['weight_license'])) for c in candidates]
    data=dict(schema_version=1,summary=dict(
        conclusion='最推荐本域人工偏好 head＋Audiobox PQ 质量约束；当前先做影子评分，尚不能直接作为已验证的 RL 自然度目标。',
        rl_ready=False,evidence='25条固定10s片段，3个AI评估器与信号诊断已实跑；人工标签为0，本轮没有新RL训练。',
        scope='8条既有项目来源（含3组生成前后配对、合成歌声、伴奏）＋2条CC-BY录制参考＋12条受控变换＋3条构造信号。',
        limitations=['背景Word已完整阅读；文档中的伪影归因与检测断言仍需验证，不能直接当作事实。','短片段、少量内容和手工反例只能发现候选问题，不能估计人类一致性或泛化收益。']),
        metadata=dict(version='naturalness-20261104-v1',date='2026-11-04',human_labels_count=0,
                      report_url=REPO+'REPORT.md',code_url=REPO+'README.md',
                      reproduce_commands=['# 完整环境、固定权重、私有原件与公开MP3重测命令见repo README',
                                          'python signal_features.py --manifest "$PRIVATE_RUN/audit_manifest.json" --output "$PRIVATE_RUN/signal_results.json"',
                                          'python evaluate_labels.py --audit runs/2026-11-04/audit.json --labels listener.json --output agreement.json'],
                      background_document=dict(name='MusicGenerationLLM_Discussion_Summary.docx',status='fully_read'),
                      limitations=['当前为已有公开模型的代理样本，未验证其代表目标连续隐状态解码架构；不能宣称目标系统已改善。',
                                   '分数绑定压缩前评分片段；播放器为独立MP3副本。两条录制参考的下载源本身是OGG，未称为原始录音母带。',
                                   'SongEval Naturalness 指人声呼吸/分句；Clarity 指结构清晰，均不是全混音digital-artifact专用检测。',
                                   '人工评分只保存在本机浏览器，默认留空；导出后才能分析一致性。']),
        samples=samples,methods=methods,comparisons=comparisons,resources=resources,
        recommendations=[
            dict(name='1 本域缺陷偏好 head（首选路线，尚未训练）',
                 formula='u(x)=wᵀ L2Norm(frozen_embedding(x)); L=−log σ(u(较少不想要人工感)−u(较多人工感))＋L2正则',
                 calibration='收集真实同prompt偏好；歌曲/prompt/生成模型分组留出。当前没有已训练权重或自然度概率。',
                 aggregation='未来完整曲用10s窗/5s hop，示例0.7mean+0.3Q20；当前只测单个10s片段，尾段时长加权尚未实现。',
                 weight='当前λ=0（只记录分数，不改变现有训练）。过听测后再探索0.02/0.05/0.1，不是已选最优权重。',
                 integration='现有reward保持运行。reward_adapter.py提供显式重标尺示例，尚未接入或训练。',
                 risk='学习电子风格/模型来源捷径、偏好简单音色，伤害合成器、配器丰富度、音乐性。'),
            dict(name='2 Audiobox PQ＋有效性和独立音乐性约束',formula='q=PQ（高分为制作质量）；整条静音/截短/失败先拒绝。PQ不更名为自然度。',
                 normalization='冻结独立校准集p05/p95映射[-1,1]；不可每batch重标尺。阈值由baseline和真实听感决定。',
                 risk='静音PQ6.735；16k mono前端看不到原始8k以上细节与空间问题。PC/CE不等于音乐性，另报SongEval和人耳。'),
            dict(name='3 PAM＋DSP仅作候选诊断',formula='官方clean/noisy-artifacts prompt softmax均值；本轮7s重叠窗[0,7]/[3,10]。',
                 risk='内容/音量/明亮度影响；自拟artifact prompt严重饱和已淘汰。DSP只解释，尚无自然度加权总分。',
                 next_step='先标两组AM与等RMS gain、正常/明亮synth、低通与失真，再听三组生成前后结果。')],
        references=[dict(title=c['name'],url=c.get('paper') or c.get('code'),code=c.get('code'),weights=c.get('weights'),note=c['role']) for c in candidates],
        provenance=read('learned_scorers_provenance.json'),
        label_agreement=dict(status='no_human_labels',spearman=None,pair_accuracy=None),
        run_status='scorer_and_perturbation_audit_complete_human_and_target_domain_validation_pending')
    (RUN/'audit.json').write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    for source,target in [('learned_results.json','learned-scores.json'),('songeval_results.json','songeval-scores.json'),('signal_results.json','signal-scores.json'),('learned_scorers_provenance.json','model-provenance.json')]:
        shutil.copyfile(work/source,RUN/target)
    buf=io.StringIO(); writer=csv.writer(buf,lineterminator='\n')
    writer.writerow(['id','kind','PQ','PC','CE','CU','PAM','failed_artifact_probe','SongEval_Naturalness','SongEval_Musicality','rms_dbfs','flatness','modulation_20_100','artificialness_human','quality_human','musicality_human'])
    for r in samples:
        writer.writerow([r['id'],r['kind'],*[r['audiobox'][k] for k in ['PQ','PC','CE','CU']],r['pam']['score'],r['pam']['artifact_score'],r['songeval']['Naturalness'],r['songeval']['Musicality'],r['signal']['rms_dbfs'],r['signal']['flatness'],r['signal']['envelope_modulation_20_100_share'],'','',''])
    (RUN/'sample-scores.csv').write_text(buf.getvalue())
    print('Joined 25 exact file hashes; raw scores, public snapshot and empty human columns written')


if __name__=='__main__': main()
