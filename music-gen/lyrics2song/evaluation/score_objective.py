"""Use unmodified frozen WSB workers for local checkpoint/CMI extensions.

The official WSB entry point rejects adapted weights and non-WSB prompts.
This wrapper keeps that entry point intact; it never claims paper acceptance.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from generate_objective import cmi_conditions, sha, write


def select_candidates(rows):
    def key(row):
        per = row['metrics']['per']
        repeat = per['asr_repeats'][per['selected_asr_repeat']]
        return per['per'], per['edits'], abs(len(repeat['phonemes']) - per['reference_phonemes']), row['candidate_index']
    indexes = sorted({r['prompt_index'] for r in rows})
    return [min((r for r in rows if r['prompt_index'] == i), key=key) for i in indexes]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--split', type=Path)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--end', type=int, required=True)
    p.add_argument('--extras-python', type=Path)
    p.add_argument('--songeval', type=Path)
    a = p.parse_args()
    a.bundle = a.bundle.resolve()
    sys.path[:0] = [str(a.bundle), str(a.bundle / 'src'), str(a.bundle / 'vendor')]
    import wsb_eval as frozen
    from mot_infer import extended_eval
    from yue2.storage import verify_result
    protocol = json.loads((a.run / 'protocol.json').read_text())
    if sha(a.source) != protocol['source_sha256']:
        raise ValueError('Source changed after generation')
    dataset = protocol['dataset']
    targets = {r['prompt_index']: r for r in frozen.read_jsonl(a.bundle / 'data/evaluation_targets.jsonl')}
    cmi = cmi_conditions(a.source, json.loads(a.split.read_text())) if dataset == 'cmi' else None
    rows = []
    report = a.run / 'evaluation'
    report.mkdir(exist_ok=True)
    config = json.loads((a.bundle / 'configs/evaluation.json').read_text())
    release = frozen.verify_release()
    metrics = ['per', 'songbench', 'mulan', 'allmusiccaps'] if dataset == 'wsb' else ['songbench', 'mulan', 'allmusiccaps']
    hashes = frozen.verify_assets(metrics)
    write(report / 'protocol.json', {'generation_protocol': protocol, 'code_files': release['code_files'],
          'asset_hashes': hashes, 'config': config, 'extension_not_official_acceptance': True,
          'songbench_runtime_extension_sha256': sha(Path(__file__).with_name('score_songbench_blackwell.py')),
          'selection': 'WSB lower-PER of 2, 4 ASR passes; CMI unselected mean of both candidates',
          'cmi_per': 'n/a: complete multilingual cohort lacks validated multilingual G2P; no English G2P for non-English lyrics',
          'q3o': 'CMI n/a: no frozen target weights; WSB pending 80GB scorer'})
    for i in range(a.start, a.end):
        for j in range(2):
            seed = ([831001, 831019] if dataset == 'wsb' else [931001, 931019])[j] + i
            ident = f'{dataset}_{i:03d}_r{j}_s{seed}'
            folder = a.run / 'songs' / ident
            result = verify_result(folder)
            binding = json.loads((folder / 'evaluation_binding.json').read_text())
            request = json.loads((folder / 'request.json').read_text())
            t = targets[i] if dataset == 'wsb' else cmi[i]
            if binding['adapter_sha256'] != protocol['adapter_sha256'] or binding['audio_sha256'] != sha(folder / 'audio.flac'):
                raise ValueError('Generation binding changed')
            if binding['result_sha256'] != sha(folder / 'result.json') or result['weights']['vae']['files']['model.safetensors']['sha256'] != frozen.SCORING_VAE_SHA:
                raise ValueError('Generation result / benchmark decoder changed')
            if request['style'] != t['style'] or request['lyrics'] != t['lyrics'] or request['seed'] != seed:
                raise ValueError('Audio does not bind the evaluated prompt')
            mp3 = report / 'audio' / ident / 'audio.mp3'
            encoding = mp3.with_suffix('.encoding.json')
            if not mp3.exists():
                mp3.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run([str(a.bundle / 'bin/ffmpeg'), '-nostdin', '-hide_banner', '-loglevel', 'error',
                    '-i', str(folder / 'audio.flac'), '-codec:a', 'libmp3lame', '-b:a', '192k', '-ar', '48000', str(mp3)], check=True)
                write(encoding, {'source_sha256': binding['audio_sha256'], 'mp3_sha256': sha(mp3)})
            encoded = json.loads(encoding.read_text())
            if encoded['mp3_sha256'] != sha(mp3) or encoded['source_sha256'] != binding['audio_sha256']:
                raise ValueError('Changed scoring MP3')
            target = {k: t[k] for k in ('control_prompt', 'tags_full', 'genre_raw', 'style_target', 'reference_lyrics')} if dataset == 'wsb' else {
                'style_target': t['style'], 'reference_lyrics': extended_eval.clean_reference(t['lyrics']),
                'control_prompt': t['lyrics'], 'tags_full': t['style'], 'genre_raw': ''}
            rows.append({'id': ident, 'prompt_index': i, 'candidate_index': j, 'seed': seed,
                'lang': t.get('lang', 'multilingual'), 'cot': 'full', 'target': target,
                'audio_path': str(mp3.resolve()), 'audio_sha256': sha(mp3), 'source_flac_sha256': binding['audio_sha256'],
                'adapter_sha256': protocol['adapter_sha256'], 'generation_truncated': result['truncated'],
                'seconds': result['audio_seconds'], 'metrics': {}})
    for metric in metrics:
        # PER is scored on both WSB candidates. Every remaining scorer sees the
        # same lower-PER candidates; CMI uses both, without quality-based cherry picking.
        scored_rows = rows if metric == 'per' or dataset == 'cmi' else select_candidates(rows)
        code = release['code_files']
        if metric == 'songbench':
            code = {**code, 'blackwell_extension': sha(Path(__file__).with_name('score_songbench_blackwell.py'))}
        jobs = []
        for row in scored_rows:
            key = extended_eval.identity(row, metric, config, hashes, code)
            path = report / 'receipts' / row['id'] / (metric + '.json')
            if extended_eval.read_cached(path, key) is None:
                jobs.append({**row, 'identity': key, 'result_path': str(path.resolve())})
        jobfile = report / (metric + '.jobs.json')
        write(jobfile, {'jobs': jobs, 'config': config})
        if jobs:
            python = a.bundle / '.venvs/metrics/bin/python'
            command = [str(python), str(a.bundle / 'wsb_eval.py'), '_worker', '--metric', metric, '--jobs', str(jobfile.resolve())]
            if metric == 'songbench':
                command = [str(python), str(Path(__file__).with_name('score_songbench_blackwell.py')),
                           '--bundle', str(a.bundle), '--jobs', str(jobfile.resolve())]
            with (report / (metric + '.log')).open('a') as log:
                subprocess.run(command, env=frozen.environment(report.resolve()), stdout=log, stderr=subprocess.STDOUT, check=True)
        for row in scored_rows:
            key = extended_eval.identity(row, metric, config, hashes, code)
            value = extended_eval.read_cached(report / 'receipts' / row['id'] / (metric + '.json'), key)
            if not value or value['status'] != 'complete':
                raise ValueError(f'{metric} missing/failed for {row["id"]}')
            row['metrics'][metric] = frozen.canonical_per(value) if metric == 'per' else frozen.canonical_songbench(value) if metric == 'songbench' else value
    selected = select_candidates(rows) if dataset == 'wsb' else rows
    write(report / f'chunk_{a.start:03d}_{a.end:03d}.json', {'complete': True, 'rows': selected})
    if a.extras_python:
        if not a.songeval:
            p.error('Supplementary scores require --songeval')
        subprocess.run([str(a.extras_python), str(Path(__file__).with_name('score_supplementary.py')),
            '--chunk', str(report / f'chunk_{a.start:03d}_{a.end:03d}.json'),
            '--songeval', str(a.songeval)], check=True)
    print(f'{protocol["arm"]} {dataset} scored conditions {a.start}:{a.end}', flush=True)


if __name__ == '__main__':
    main()
