"""Publish verified manual recovery, with failed/missing evaluations kept explicit."""
import csv
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
PUBLIC=HERE.parents[2]/'platform/site/demos/weekly-2026-10-23'
source=HERE/'recovery-2026-10-26.json'
record=json.loads(source.read_text())
lab=record['machines']['lab5090'];gadi=record['machines']['gadi']
jobs=[
 dict(name='Muse · Mean7 · LR3e-4',resource='5090',status='270步与四个valid checkpoint已核验',progress='120/120 valid；1080/1080 train已处理，0/50步含15/3条不可评分；test失败，不报局部均值'),
 dict(name='Muse · SP Richness · LR3e-4 · v2',resource='5090',status='270步完成，替代旧轮展示',progress='0/50/100/270各30/30 valid；旧轮原始存档保留'),
 dict(name='DiffRhythm2 · Mean7 · LR1e-4',resource='5090',status='270步与120/120 valid已核验',progress='与3e-4共享逐音频SHA一致的冻结baseline；原生60s预算'),
 dict(name='Muse · Mean7 · LR1e-4',resource='Gadi iv96 · 182084203',status='训练完成 · 验证失败',progress='270/270更新、540音频绑定、最终checkpoint已核验；FlashInfer JIT要求GCC9+，无valid均值'),
 dict(name='YuE2 · Mean7 · LR3e-4',resource='Gadi wa66 · 182083826',status='采样验收失败 · 0更新',progress='原生NAR/PEFT路径缺rotary_emb；没有新Mean7正式结果'),
 dict(name='Muse · WildSongBench test',resource='5090',status='评估失败 · 遗留子进程未退出',progress=f'baseline生成{lab["test"]["generated"]}/384，其中{lab["test"]["missing_audio_n"]}条无音频；0份完整评分receipt，不显示均值'),
 dict(name='Kaya旧评测任务',resource='Kaya',status='用户暂停 · 未解除',progress=f'{record["machines"]["kaya"]["held_n"]}个JobHeldUser；无新增运行或新结果'),
]
snapshot=dict(checked_at=record['checked_at'],protocol='CMI270 train /30 valid；WildSongBench192×2 test。手动回收，不是实时监控；没有启动训练、解锁或定时任务。',
 jobs=jobs,traces=[gadi['muse']],technical_invalid_policy='缺失七维保持null；训练utility0不当作SongBench零分。',
 receipt_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
for folder in [HERE,PUBLIC]:
 (folder/'next-experiments.json').write_text(json.dumps(snapshot,indent=2,allow_nan=False)+'\n')
 (folder/'recovery-2026-10-26.json').write_text(source.read_text())
 with (folder/'cohort-results.csv').open('w',newline='') as f:
  writer=csv.writer(f,lineterminator='\n')
  writer.writerow(['model','reward','lr','split','step','processed_n','scored_n','SongBench_mean7','invalid_n','status','receipt_sha256'])
  for row in lab['cohorts']:
   writer.writerow([row[k] if row[k] is not None else '' for k in ['model','reward','lr','split','step','n','scored_n','mean7','invalid_n','status','receipt_sha256']])
print('Published7 current job states,1 training-only trace,4 complete cohort processing receipts; no partial SongBench mean.')
