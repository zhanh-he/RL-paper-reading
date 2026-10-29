"""Publish genuine CPU controls and planned/queued model audits, never invented scores."""
import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import mido
import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT/'platform/site'
RUN = 'music-trans/reliability/runs/2026-10-29'
URL = 'https://github.com/zhanh-he/RL-paper-reading/blob/main/'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def midi_audio(notes, name, destination):
    destination.mkdir(parents=True, exist_ok=True)
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack(); midi.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo', tempo=500000))
    events = []
    sr = 48000
    wave = np.zeros(11*sr, dtype=np.float32)
    for note in notes:
        for when, kind in [(note['start'],'note_on'),(note['end'],'note_off')]:
            events.append((round(when*960),kind,note['pitch']))
        start = round(note['start']*sr)
        t = np.arange(round((note['end']-note['start'])*sr))/sr
        envelope = np.minimum(1,t/.01)*np.minimum(1,(t[-1]-t)/.02)
        wave[start:start+len(t)] += (.2*envelope*np.sin(2*np.pi*440*2**((note['pitch']-69)/12)*t)).astype(np.float32)
    last = 0
    for tick, kind, pitch in sorted(events):
        track.append(mido.Message(kind,note=pitch,velocity=80 if kind=='note_on' else 0,time=tick-last)); last=tick
    midi.save(destination/f'{name}.mid')
    with tempfile.TemporaryDirectory(prefix='amt-metric-audio-') as tmp:
        original = Path(tmp)/'scoring-independent-preview.wav'
        wavfile.write(original,sr,wave)
        original_hash = sha(original)
        subprocess.run(['ffmpeg','-v','error','-y','-i',str(original),'-codec:a','libmp3lame','-b:a','192k','-ar','48000',str(destination/f'{name}.mp3')], check=True)
    asset = dict(notes=notes,tracks=['Artificial melody'],midi=f'amt/demos/controls/{name}.mid',audio=f'amt/demos/controls/{name}.mp3')
    return asset, dict(name=name, midi_sha256=sha(destination/f'{name}.mid'), preview_wav_sha256=original_hash,
                       public_mp3_sha256=sha(destination/f'{name}.mp3'), render='sine; 48kHz; MP3 192kbps; no loudness normalization; preview only, not scored audio')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric-result', type=Path, required=True)
    parser.add_argument('--job-id', required=True)
    args = parser.parse_args()
    original = json.loads(args.metric_result.read_text())
    owned = ROOT/RUN; owned.mkdir(parents=True,exist_ok=True)
    (owned/'metric-controls.json').write_bytes(args.metric_result.read_bytes())
    fields = [('class_onset','Pitch-class onset F1','pitch_class_onset_f1_percent'),('abs_onset','Absolute onset F1','absolute_pitch_onset_f1_percent'),('abs_onoff','Absolute onset+offset F1','absolute_pitch_onset_offset_f1_percent')]
    definitions = [dict(id=key,label=label,unit='fraction') for key,label,_ in fields]
    condition_labels = ['不变 / Control','整段高八度','时值翻倍','交替八度错误','整体延后 100ms']
    receipts=[]; conditions=[]
    for i,row in enumerate(original['rows']):
        notes=[dict(track=0,pitch=int(p),start=float(interval[0]),end=float(interval[1])) for p,interval in zip(row['estimated_midi_pitches'],row['estimated_intervals_seconds'])]
        asset, receipt=midi_audio(notes,row['case'],SITE/'amt/demos/controls');receipts.append(receipt)
        conditions.append(dict(id=row['case'],label=condition_labels[i],status='measured',outputs={'note-control':asset},evaluation=dict(cohort='one-artificial-note-sequence',n=1,values={key:row[field]/100 for key,_,field in fields})))
    (owned/'preview-hashes.json').write_text(json.dumps(receipts,indent=2)+'\n')
    base=dict(kind='evaluation',stages=[],trace=[],reward='Evaluation only; no optimization',method='Frozen evaluation; no parameter updates')
    e0=dict(base,id='metric-blindspots',title='E0 · 评分正负控制（不是模型转录）',status='measured',task='Artificial-note scorer audit',model='mir_eval 0.8.2; no AMT model',receipt=RUN+'/README.md',
        evaluation_protocol='One artificial sequence of 12 notes. F1 uses 50ms onset / 50 cents; offset tolerance max(50ms,20% ref duration). Pitch-class folding is explicit. MIDI/MP3 are newly rendered listening previews, not model outputs or scored audio.',
        samples=[dict(id='note-control',title='12 个人工音符 · 非真实歌曲',duration=11.,reference=conditions[0]['outputs']['note-control'],note='Only artificial note arrays were scored. Public MIDI/audio are sine-wave listening illustrations created after scoring; no transcription model ran in E0.')],
        conditions=conditions,metric_definitions=definitions)
    reference=440*2**((np.array(original['reference_midi_pitches'])-69)/12)
    f0rows=[]
    for name,estimate in [('unchanged',reference.copy()),('octave-up',reference*2),('plus-100hz',reference+100),('reverse-contour',reference[::-1])]:
        correlation=float(np.corrcoef(reference,estimate)[0,1])
        cents=float(np.mean(np.abs(1200*np.log2(estimate/reference))))
        f0rows.append(dict(id=name,label=name,status='measured',evaluation=dict(cohort='one-artificial-voiced-f0-sequence',n=1,values={'fpc':correlation,'cents':cents})))
    assert abs(f0rows[1]['evaluation']['values']['fpc']-1)<1e-9
    assert abs(f0rows[1]['evaluation']['values']['cents']-1200)<1e-6
    (owned/'f0-controls.json').write_text(json.dumps(dict(scope='CPU mathematical control; not RMVPE or YingMusic inference',reference_hz=reference.tolist(),rows=f0rows),indent=2)+'\n')
    f0=dict(base,id='f0-correlation-controls',title='E0b · F0 相关性盲区（非生成模型）',status='measured',task='Pitch-correlation reward sanity control',model='NumPy Pearson correlation; no singer model',receipt=RUN+'/README.md',samples=[],conditions=f0rows,
        evaluation_protocol='One nonconstant artificial voiced F0 sequence; no unvoiced frames, alignment search or model inference. Pearson can be negative. Cents error uses absolute frequencies. A 2x frequency shift preserves Pearson=1 but is 1200 cents wrong.',
        metric_definitions=[dict(id='fpc',label='F0 Pearson r',unit='number'),dict(id='cents',label='Mean absolute cents ↓',unit='number')])
    gpu=dict(base,id='sheetsage2-harmonic-smoke',title='E1 · SheetSage2 冻结模型 smoke / 微扰审计',status='queued',task='Raw procedural harmonic audio -> timed melody events',model='SheetSage2 b83c3d79 + MERT parent d8ba1c74',receipt=RUN+'/sheetsage2.md',samples=[],
        evaluation_protocol=f'Slurm job {args.job_id}; one allocated V100-32GB, 30-minute cap, at most 48 attempts. 12 procedural cases x four conditions. Not singing and not independent works. Default 300s window retained; paper preset; FP32 if V100 lacks BF16. No measured model-quality scores yet.',
        metric_definitions=definitions,conditions=[dict(id=c,label=l,status='planned') for c,l in [('original','原版'),('prefix-500ms','前置静音 .5s'),('gain-minus6db','增益 -6dB'),('noise-snr30','30dB SNR 噪声')]])
    future=dict(base,id='sheetsage2-real-vocal-pilot',title='E2 · 真实歌声准确率 / 上下文审计',status='planned',task='Licensed real vocal melody evaluation',model='SheetSage2-AR; frozen comparators subject to weight/data access',receipt=RUN+'/PLAN.md',samples=[],metric_definitions=definitions,
        evaluation_protocol='No cohort is claimed yet. Freeze licensed audio, song-disjoint dev/test and annotation versions before scoring. Compare crop/full context, absolute octave/offset and raw/separated vocals; do not tune shifts on test.',
        conditions=[dict(id=c,label=l,status='planned') for c,l in [('original','原始混音'),('crop','裁剪 / 上下文'),('separated','冻结分离器前端')]])
    parser_result=json.loads((owned/'ying-midi-interface.json').read_text())
    interface=dict(base,id='ying-midi-interface',title='E0c · YingMusic MIDI 接口实测（非生成）',status='measured',task='Pinned upstream scalar MIDI parser control',model='YingMusic V1 get_notemidi_from_file; no neural model',receipt=RUN+'/ying-midi-interface.json',samples=[],
        metric_definitions=[dict(id='lead_fraction',label='Retained lead frames',unit='fraction')],
        evaluation_protocol='CPU execution of the exact pinned upstream parser only. One-second overlapping notes 72 and 60, identical musical content with reversed track order. No singer model or weights. This is an interface counterexample, not a dataset mean or singing accuracy score.',
        conditions=[dict(id=row['id'],label=label,status='measured',evaluation=dict(cohort='one-overlapping-note-interface-control',n=1,values={'lead_fraction':row['retained_lead_fraction']})) for row,label in zip(parser_result['rows'],['仅 lead 72','lead 72 → acomp 60','acomp 60 → lead 72'])])
    control=dict(base,id='melody-control-comparison',title='E3 · 同一 YingMusic V1 的旋律提取链对照',status='planned',task='Extractor/pipeline comparison; not continuous-F0 vs full score',model='YingMusic V1 6d7c4537 / SheetSage2; no generation runs yet',receipt=RUN+'/PLAN.md',samples=[],metric_definitions=[],
        evaluation_protocol='Planned 8 licensed monophonic excerpts x 3 arms x 3 seeds, not a claimed cohort. V1 audio route uses RMVPE voicing + SOME notes; MIDI uses scalar notes. Verify time grids / reference boundary first. Same vocal input, frozen generator, lyrics/timbre/seed/NFE/CFG/pitch shift/duration. Monophonic lead only; chord/polyphonic information is discarded. Cross-generator output is only a system comparison.',
        conditions=[dict(id=c,label=l,status='planned') for c,l in [('native','原生 SOME 音符条件'),('score-midi','SheetSage2 vocal MIDI'),('oracle','人工标注 MIDI 参考')]])
    path=ROOT/'platform/data/amt/experiments.json'; manifest=json.loads(path.read_text())
    updates=[e0,f0,interface,gpu,future,control]; ids={r['id'] for r in updates}
    manifest['experiments']=[r for r in manifest['experiments'] if r['id'] not in ids]+updates
    manifest['updated_at']='2026-10-29'
    manifest['reported_results']=[dict(name=name,rwc=rwc,rock=rock,output=output,source='https://arxiv.org/html/2610.05336v1#A4.SS3') for name,rwc,rock,output in [
        ('SheetSage2-Prober',83.08,65.98,'vocal melody'),('SheetSage2-AR',82.51,67.08,'vocal melody'),
        ('SheetSage1',62.71,49.19,'melody vs vocal reference'),('MuScriptor large',47.04,37.36,'voice class from full transcription'),
        ('YourMT3+',49.12,36.05,'singing-voice / melody class'),('HT Demucs + ROSVOT',26.09,19.86,'separated vocal, <=20s chunks')]]
    for collection,records in {
        'datasets':[dict(name='E0 artificial controls',labels='12 intended notes; synthetic F0; no singing labels',split='Deterministic control, not held-out music',source=URL+RUN+'/README.md'),dict(name='Procedural harmonic smoke 12',labels='Original harmonic signals and intended notes',split='12 related procedural cases; no real-world generalization claim',source=URL+'music-trans/reliability/prepare_smoke.py')],
        'models':[dict(name='SheetSage2-AR',interface='audio -> ABC / timed note events / MIDI',checkpoint='b83c3d79; parent d8ba1c74; frozen; GPU audit submitted',source='https://huggingface.co/m-a-p/SheetSage2'),dict(name='YingMusic-Singer',interface='V1: RMVPE voicing + SOME frame notes, or monophonic MIDI -> note indices -> singing',checkpoint='V1 6d7c4537; CPU MIDI parser tested; singer weights not run; original paper differs',source='https://github.com/GiantAILab/YingMusic-Singer')],
        'rewards':[dict(name='Strict note audit',definition='50ms onset; absolute pitch; optional max(50ms,20%) offset',failure_case='Pitch-class-only F1 ignores octaves and note lengths',source=URL+RUN+'/metric-controls.json'),dict(name='F0 Pearson diagnostic',definition='Voiced F0 Pearson correlation, not absolute pitch accuracy',failure_case='F0 x2 keeps r=1 while adding one octave error',source=URL+RUN+'/f0-controls.json')]
    }.items():
        new_names={r['name'] for r in records}
        manifest[collection]=[r for r in manifest[collection] if r['name'] not in new_names]+records
    path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'published_controls':3,'model_audits':3,'gpu_job_id':args.job_id,'audio_hashes':len(receipts)}))


if __name__ == '__main__':
    main()
