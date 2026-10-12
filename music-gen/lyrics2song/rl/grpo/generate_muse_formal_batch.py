"""Native vLLM fixed-seed Muse batch replays, with separately decoded receipts."""

import argparse
import hashlib
import json
from pathlib import Path

VALID_SHA = 'ed5c5f36af48ef6ce81d15c7b15b429fb250ec4a52a04a43e55d96ec3cafb59d'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def groups(study):
    yield study / 'native_vllm/long_baseline', None
    for reward in ('musecritic', 'songeval'):
        for lr in ('1e-6', '3e-6', '1e-5'):
            run = study / f'{reward}_lr{lr}_train_v2'
            verified = json.loads((run / 'verified_recovery.json').read_text())
            if verified['verified_train_steps'] < 1:
                raise ValueError(f'Unaudited source: {run}')
            for step in sorted(verified['checkpoints'], reverse=True):
                if step not in (30, 60, 90, 120, 270):
                    continue
                matches = list(run.glob(f'checkpoint/v*/checkpoint-{step}/adapter_model.safetensors'))
                if len(matches) != 1:
                    raise ValueError(f'Expected one adapter at {run}, step {step}')
                yield run / f'long_replay_vllm/step_{step:06d}/grpo', matches[0].parent


def generate(args):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    import vllm

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    start = tokenizer.convert_tokens_to_ids('<AUDIO_0>')
    end = tokenizer.convert_tokens_to_ids('<AUDIO_16383>')
    if end - start != 16383:
        raise ValueError('Noncontiguous Muse audio vocabulary')
    records = [json.loads(line) for line in args.valid.read_text().splitlines()]
    if len(records) != 30:
        raise ValueError('Expected 30 validation prompts')
    engine = LLM(model=str(args.model), trust_remote_code=True, dtype='bfloat16',
                 enforce_eager=True, max_model_len=4096, gpu_memory_utilization=0.55,
                 max_num_seqs=3, enable_lora=True, max_lora_rank=8)
    for adapter_id, (folder, adapter) in enumerate(groups(args.study), 1):
        adapter_sha = sha(adapter / 'adapter_model.safetensors') if adapter else None
        paths, prompts, params, indexes = [], [], [], []
        for index in (0, 10, 20):
            path = folder / f'valid_{index:02d}/audio.json'
            expected = {'seed': 5101 + index, 'max_new_tokens': 1600, 'force_length': True,
                        'backend': 'vllm', 'vllm_version': vllm.__version__, 'dtype': 'bfloat16',
                        'adapter_sha256': adapter_sha, 'validation_jsonl_sha256': VALID_SHA}
            if path.exists():
                stored = json.loads(path.read_text())
                if any(stored.get(k) != v for k, v in expected.items()):
                    raise ValueError(f'Changed generation configuration: {path}')
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            messages = [records[index]['messages'][0]]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt); paths.append((path, expected)); indexes.append(index)
            params.append(SamplingParams(seed=5101 + index, max_tokens=1600, min_tokens=1600,
                                         temperature=0.9, top_p=0.9, repetition_penalty=1.3))
        if not paths:
            continue
        request = LoRARequest(f'formal-{adapter_id}', adapter_id, str(adapter)) if adapter else None
        outputs = engine.generate(prompts, params, lora_request=request, use_tqdm=False)
        if len(outputs) != len(paths):
            raise ValueError('Batch generation lost outputs')
        for (path, expected), index, output in zip(paths, indexes, outputs):
            ids = list(output.outputs[0].token_ids)
            row = {**expected, 'index': index, 'generated_ids': ids,
                   'audio_tokens': [value - start for value in ids if start <= value <= end]}
            path.write_text(json.dumps(row) + '\n')
            if len(row['audio_tokens']) < 32:
                (path.parent / 'failure.json').write_text(json.dumps({**expected,
                    'index': index, 'reason': 'Fewer than 32 AUDIO tokens; no fabricated audio.',
                    'audio_token_count': len(row['audio_tokens']), 'generated_ids_sha256': sha(path)}, indent=2) + '\n')
            print(f'{path}: {len(row["audio_tokens"])} audio tokens', flush=True)


def finalize(args):
    import soundfile as sf
    from muse_songeval_pilot import signal

    for folder, adapter in groups(args.study):
        for index in (0, 10, 20):
            root = folder / f'valid_{index:02d}'
            tokens = json.loads((root / 'audio.json').read_text())
            if (root / 'failure.json').exists():
                print(f'Unrenderable output retained: {root}', flush=True)
                continue
            wav = root / 'audio.wav'
            if not wav.exists():
                raise FileNotFoundError(f'No decoded audio: {root}')
            receipt = root / 'receipt.json'
            flac = root / 'audio.flac'
            if receipt.exists():
                if json.loads(receipt.read_text())['flac_sha256'] != sha(flac):
                    raise ValueError('Archived native audio changed')
                continue
            audio, rate = sf.read(wav, dtype='float32')
            if rate != 48000:
                raise ValueError('Unexpected MuCodec sample rate')
            sf.write(flac, audio, rate, subtype='PCM_24')
            record = {k: tokens[k] for k in ('index', 'seed', 'backend', 'vllm_version', 'dtype',
                                             'force_length', 'adapter_sha256', 'validation_jsonl_sha256')}
            record.update(max_tokens=1600, seconds=len(audio) / rate,
                          audio_tokens=len(tokens['audio_tokens']), truncated=True,
                          decoder_window_seconds=40.96, decoder_overlap_native=True,
                          wav_sha256=sha(wav), flac_sha256=sha(flac), signal=signal(flac),
                          reaches_30_seconds=len(audio) / rate >= 30)
            receipt.write_text(json.dumps(record, indent=2) + '\n')
            print(f'{root}: {record["seconds"]:.2f} seconds', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('generate', 'finalize'))
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--valid', type=Path, required=True)
    args = parser.parse_args()
    if sha(args.valid) != VALID_SHA:
        raise ValueError('Frozen validation projection changed')
    (generate if args.phase == 'generate' else finalize)(args)


if __name__ == '__main__':
    main()
