"""Bind a curated real-Suno shortlist and full-resolution traces to frozen audio."""
import argparse,hashlib,json
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--audit',required=True);p.add_argument('--features',required=True);p.add_argument('--fulltrace',required=True);p.add_argument('--selection',required=True);p.add_argument('--fulltrace-code',default=str(Path(__file__).resolve().parent/'auditory_fulltrace.py'));p.add_argument('--output',required=True);a=p.parse_args()
 read=lambda p:json.loads(Path(p).read_text());audit=read(a.audit);old={r['id']:r for r in read(a.features)['samples']};full=read(a.fulltrace);selected=read(a.selection);samples={r['id']:r for r in audit['samples']};traces=[];maxdiff=0
 assert full['status']=='complete';assert len(full['samples'])==len(samples)==48
 for r in full['samples']:
  base=old[r['id']];sample=samples[r['id']];assert r['audio_sha256']==base['audio_sha256']==sample['excerpt_sha256']
  for key in ['modulation_depth_20_300','isolated_resonance_proxy','harmonic_veto_resonance_proxy']:
   diff=abs(r['auditory'][key]-base['auditory'][key]);assert diff<1e-6;maxdiff=max(maxdiff,diff)
  t=r['auditory']['resonance_time'];assert len(t['time_s'])==len(t['harmonic_veto_excess'])>900
  traces.append(dict(id=r['id'],audio_sha256=r['audio_sha256'],time_s=t['time_s'],value=t['harmonic_veto_excess']))
 pairs=[]
 for item in selected['pairs']:
  checkpoints=[]
  for side in ['a','b']:
   s=item[side];t=next(r for r in traces if r['id']==s['id']);ix=max(range(len(t['value'])),key=lambda i:t['value'][i])
   for method,series in [('modulation',samples[s['id']]['timeline']['modulation']),('roughness',samples[s['id']]['timeline']['roughness']),('resonance',dict(time_s=t['time_s'],value=t['value']))]:
    peak=max(range(len(series['value'])),key=lambda i:series['value'][i]);clip=series['time_s'][peak]
    if series['value'][peak]<=0:continue
    checkpoints.append(dict(sample_id=s['id'],clip_s=clip,source_s=s['source_start_s']+clip,method=method,note='Nominal analysis-window center, not confirmed artifact onset.'))
  aa,bb=item['a']['metrics'],item['b']['metrics']
  obs=f"子带调制 {aa['modulation_depth']:.4f}/{bb['modulation_depth']:.4f}；DW70 {aa['roughness70']:.4f}/{bb['roughness70']:.4f} asper；保护共振 {aa['resonance_protected']:.5f}/{bb['resonance_protected']:.5f}。"
  pairs.append(dict(id=f'suno-listen-{item["priority"]}',label=f'精选{item["priority"]:02d} · '+item['title'],a=item['a']['id'],b=item['b']['id'],purpose=item.get('purpose',item['title']),observation=obs,caveat=item.get('caveat',item.get('limitations','同族/不同段落并非严格同输入；没有人工缺陷标签。')),checkpoints=checkpoints))
 data=dict(schema_version=1,metadata=dict(date='2026-11-05',version='suno-listening-review-20261105-v1',description='先听4组真实Suno；第5组为同曲不同段落的局部粗糙峰。按指标分歧与保护对照探索性挑选，不能用本精选集报留出检测准确率。',trace_note='补算10ms共振轨迹修复旧100ms点抽样漏峰；原音频与原标量分数不变。峰值为分析窗位置，不是已确认缺陷起点。',human_labels_count=0,report_url='https://github.com/zhanh-he/RL-paper-reading/blob/main/music-gen/reward-regime-study/naturalness/LISTENING_REVIEW_2026-11-05.md'),source_audit_sha256=sha(a.audit),source_feature_sha256=sha(a.features),fulltrace_run_profile=full['profile'],checks=dict(same_audio_hashes=True,maximum_scalar_difference=maxdiff,traces_count=len(traces),human_labels_count=0,selection_is_heldout_validation=False,fulltrace_source_file_sha256=sha(a.fulltrace),fulltrace_code_sha256=sha(a.fulltrace_code)),shortlist=pairs,traces=traces)
 Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(data,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n');print('Bound48 corrected traces and5 exploratory realSuno pairs; unchanged waveform/scalar hashes.')
if __name__=='__main__':main()
