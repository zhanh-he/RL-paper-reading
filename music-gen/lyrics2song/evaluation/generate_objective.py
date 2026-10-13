"""Resumable, full-song test generation. Never trains or selects on test scores."""

import argparse
import hashlib
import json
import time
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def cmi_conditions(source, split):
    # Use public split IDs, not row count / preference votes as the sampling unit.
    votes = [json.loads(line) for line in source.read_text().split('\n') if line.strip()]
    by_id = {str(v['prompt id']): v for v in votes}
    result = []
    for entry in split['test']:
        selected = [by_id[i] for i in entry['source_prompt_ids']]
        v = selected[0]
        normalized = lambda text: ' '.join(text.split()).casefold()
        key = (normalized(v['prompt']), normalized(v['lyrics']), v['ref-audio-path'].strip())
        cid = hashlib.sha256(json.dumps(key, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
        if cid != entry['condition_id'] or any(
            (normalized(s['prompt']), normalized(s['lyrics']), s['ref-audio-path'].strip()) != key
            for s in selected
        ):
            raise ValueError('CMI-Pref test split does not bind its source conditions')
        result.append({'condition_id': cid, 'style': v['prompt'].strip(), 'lyrics': v['lyrics'].strip()})
    if len(result) != 121:
        raise ValueError('Expected 121 frozen CMI-Pref test conditions')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['wsb', 'cmi'], required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--split', type=Path)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--vae', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--adapter', type=Path)
    p.add_argument('--arm', required=True)
    p.add_argument('--max-tokens', type=int, default=9000)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--end', type=int)
    a = p.parse_args()
    source_sha = sha(a.source)
    if a.dataset == 'wsb':
        if source_sha != '859e5225d319912efb64d587d1d5eb6cf0abc9fb162cc3d5f02d1688cff2edc0':
            raise ValueError('WildSongBench frozen source changed')
        rows = [json.loads(line) for line in a.source.read_text().split('\n') if line.strip()]
        if len(rows) != 192 or [r['prompt_index'] for r in rows] != list(range(192)):
            raise ValueError('Expected all 192 ordered WildSongBench prompts')
    else:
        if source_sha != '824a92f33d04096e739fdc771d09bda125943baafae5c0eb5b6e9e267d3762e9':
            raise ValueError('CMI-Pref frozen source changed')
        if not a.split:
            p.error('CMI test requires the frozen public split')
        if sha(a.split) != 'db769e27d75a70f2e819d6de771002652bb37e9adeb4a549fca66d9dfa30c616':
            raise ValueError('CMI-Pref split changed')
        rows = cmi_conditions(a.source, json.loads(a.split.read_text()))
    import torch
    from yue2 import YuE2Pipeline
    from yue2.storage import verify_result
    adapter_sha = sha(a.adapter / 'adapter_model.safetensors') if a.adapter else None
    protocol = {'dataset': a.dataset, 'source_sha256': source_sha, 'split_sha256': sha(a.split) if a.split else None,
                'arm': a.arm, 'adapter_sha256': adapter_sha, 'test_conditions': len(rows),
                'candidates': 2, 'cot': 'full', 'max_semantic_tokens': a.max_tokens, 'min_tokens': 200,
                'decoder': a.vae, 'decoder_revision': 'b54118f0fc462f08999d1ec07e88817f4ee3f770',
                'reference_audio_used': False, 'seed_rule': 'WSB: 831001+i / 831019+i; CMI: 931001+i / 931019+i',
                'checkpoint_selection': 'fixed final step 270; no test-dependent checkpoint selection'}
    if a.adapter:
        protocol['adapter_application'] = 'PEFT merge_and_unload into bf16 native AR/NAR model'
    a.output.mkdir(parents=True, exist_ok=True)
    target = a.output / 'protocol.json'
    if target.exists() and json.loads(target.read_text()) != protocol:
        raise ValueError('Output directory already binds a different protocol')
    write(target, protocol)
    # Merge into the native model so both AR and NAR see adapted weights.
    with YuE2Pipeline.from_pretrained(a.model, vae=a.vae,
            vae_revision=protocol['decoder_revision'], device='cuda:0', backend='torch-eager',
            memory_budget_gib=32, progress=False) as pipe:
        expected = {'mot': '1d55c42c1a9875c34f5d736e15078449992b044e807ce2a138e6cf289a1e59e9',
                    'vae': 'b6d283628913bb41145ba99e2314eef613905ee95f690eb70e8212d5f4965044'}
        for name, checksum in expected.items():
            if pipe.weights[name]['files']['model.safetensors']['sha256'] != checksum:
                raise ValueError(f'Wrong benchmark weight identity: {name}')
        if a.adapter:
            from peft import PeftModel
            wrapped = PeftModel.from_pretrained(pipe._load_model(), a.adapter, is_trainable=False)
            pipe._model = wrapped.merge_and_unload().eval()
        for i in range(a.start, min(a.end or len(rows), len(rows))):
            for j in range(2):
                seed = ([831001, 831019] if a.dataset == 'wsb' else [931001, 931019])[j] + i
                ident = f'{a.dataset}_{i:03d}_r{j}_s{seed}'
                folder = a.output / 'songs' / ident
                receipt = folder / 'evaluation_binding.json'
                if receipt.exists():
                    cached = json.loads(receipt.read_text())
                    result = verify_result(folder)
                    if cached['adapter_sha256'] != adapter_sha or cached['source_sha256'] != source_sha or cached['audio_sha256'] != sha(folder / 'audio.flac'):
                        raise ValueError(f'Changed cached evaluation identity: {ident}')
                    continue
                if folder.exists():
                    raise FileExistsError(f'Incomplete prior generation needs inspection: {folder}')
                started = time.monotonic()
                with torch.inference_mode():
                    song = pipe(id=ident, style=rows[i]['style'], lyrics=rows[i]['lyrics'],
                                cot='full', cfg_scale=1.0, seed=seed,
                                semantic_sampling={'max_tokens': a.max_tokens, 'min_tokens': 200})
                result = song.save_artifacts(folder)
                verify_result(folder)
                binding = {'id': ident, 'condition_index': i, 'candidate_index': j, 'seed': seed,
                           'adapter_sha256': adapter_sha, 'source_sha256': source_sha,
                           'audio_sha256': sha(folder / 'audio.flac'), 'result_sha256': sha(folder / 'result.json'),
                           'seconds': result['audio_seconds'], 'truncated': result['truncated'],
                           'elapsed_seconds': time.monotonic() - started}
                write(receipt, binding)
                write(a.output / 'progress.json', {'arm': a.arm, 'dataset': a.dataset, 'last': binding,
                      'completed_candidates': len(list((a.output / 'songs').glob('*/evaluation_binding.json'))),
                      'expected_candidates': len(rows) * 2})
                print(json.dumps(binding), flush=True)
                del song


if __name__ == '__main__':
    main()
