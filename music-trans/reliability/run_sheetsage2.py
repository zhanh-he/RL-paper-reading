"""Bounded raw-audio SheetSage2 audit. No training, codec or dataset download."""
import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel

SHEET = 'b83c3d79a6d98bcffaf8f9ed112c90aa887887bc'
PARENT = 'd8ba1c745e733b3908ce6ad16ebeb17ac7600a42'


def save(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=48)
    args = parser.parse_args()
    if not 1 <= args.limit <= 48:
        raise ValueError('Bounded to at most 48 inference attempts')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    state = dict(status='running', phase='loading', started_at_unix=started,
                 sheet_revision=SHEET, parent_revision=PARENT, training=False,
                 slurm_job_id=os.environ.get('SLURM_JOB_ID'), attempts=[],
                 source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    state_path = args.output/'state.json'
    save(state_path, state)
    try:
        assert torch.cuda.is_available(), 'An allocated CUDA device is required'
        state.update(gpu=torch.cuda.get_device_name(), torch=torch.__version__)
        # V100 has no native BF16; keep the paper window/preset but disclose FP32.
        dtype = 'bf16' if torch.cuda.is_bf16_supported() else 'fp32'
        state['precision'] = dtype
        state['paper_precision_match'] = dtype == 'bf16'
        save(state_path, state)
        torch.manual_seed(20261029)
        model = AutoModel.from_pretrained(str(args.assets/'SheetSage2'),
            trust_remote_code=True, local_files_only=True,
            base_model_path=str(args.assets/'MERT-v2-FullSong'), torch_dtype=torch.float32).eval().to('cuda')
        state.update(phase='inference', model_loaded=True, input_audio_length=model.config.input_audio_length)
        save(state_path, state)
        manifest = json.loads((args.inputs/'manifest.json').read_text())
        count = 0
        for sample in manifest['samples']:
            source = args.inputs/sample['npz']
            assert hashlib.sha256(source.read_bytes()).hexdigest() == sample['sha256']
            original = np.load(source, allow_pickle=False)['waveform']
            for condition in manifest['conditions']:
                if count >= args.limit:
                    break
                count += 1
                wave = original.copy()
                if condition == 'prefix-500ms':
                    wave = np.pad(wave, (12000, 0))
                elif condition == 'gain-minus6db':
                    wave *= 10**(-6/20)
                elif condition == 'noise-snr30':
                    noise = np.random.default_rng(20261029 + count).normal(size=wave.shape)
                    noise *= np.sqrt(np.mean(wave.astype(float)**2)/1000/np.mean(noise**2))
                    wave += noise.astype(np.float32)
                directory = args.output/sample['id']/condition
                record = dict(sample=sample['id'], condition=condition, status='running', start_unix=time.time())
                state['attempts'].append(record)
                save(state_path, state)
                torch.cuda.reset_peak_memory_stats()
                result = model.transcribe(wave, sampling_rate=24000, output_dir=str(directory), preset='paper', dtype=dtype)
                torch.cuda.synchronize()
                events = result.get('events', [])
                if not isinstance(events, list) or not events:
                    raise RuntimeError('No nonempty event list produced')
                record.update(status='measured', seconds=time.time()-record['start_unix'],
                              peak_allocated_bytes=torch.cuda.max_memory_allocated(), num_events=len(events),
                              abc_nonempty=bool(result.get('abc')), abc_error=result.get('abc_error'))
                save(state_path, state)
                print(json.dumps(record), flush=True)
            if count >= args.limit:
                break
        state.update(status='measured', phase='completed', finished_at_unix=time.time())
        save(state_path, state)
    except Exception as error:
        state.update(status='failed', phase='failed', error=repr(error), traceback=traceback.format_exc(), finished_at_unix=time.time())
        if state['attempts'] and state['attempts'][-1]['status'] == 'running':
            state['attempts'][-1].update(status='failed',error=repr(error))
        save(state_path, state)
        raise


if __name__ == '__main__':
    main()
