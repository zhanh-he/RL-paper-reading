"""Explicit invalid-output penalties; decoder/scorer infrastructure remains strict."""

import importlib.util
import json
import os
import sys
import threading
from pathlib import Path

import soundfile as sf
import numpy as np
from swift.rewards import orms

spec = importlib.util.spec_from_file_location(
    'muse_cmi_rewards_aligned_base', Path(__file__).with_name('muse_cmi_rewards.py'))
original = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = original
spec.loader.exec_module(original)
_lock = threading.Lock()


class InvalidCompletion(ValueError):
    pass


class InvalidOutputGuard:
    def _invalid(self, request_id, completion, reason, audio_path=''):
        record = {'request_id': request_id, 'reason': reason, 'reward': 0.0,
                  'audio_tokens': original.base._count_audio_tokens(completion),
                  'audio_path': audio_path}
        path = Path(os.environ['MUSE_INVALID_OUTPUT_LOG'])
        with _lock:
            with path.open('a') as handle:
                handle.write(json.dumps(record) + '\n')
        return {'audio_path': '', 'error': InvalidCompletion(reason)}

    def _decode_completions_to_wavs(self, items):
        results = [None] * len(items)
        valid_indices = []
        for i, (request_id, completion) in enumerate(items):
            if not original.base._count_audio_tokens(completion):
                results[i] = self._invalid(request_id, completion, 'no AUDIO tokens')
            else:
                valid_indices.append(i)
        decoded = super()._decode_completions_to_wavs([items[i] for i in valid_indices])
        if len(decoded) != len(valid_indices):
            raise RuntimeError('Decoder result count mismatch')
        minimum = float(os.environ.get('MUSE_MIN_AUDIO_SECONDS', '1'))
        for i, result in zip(valid_indices, decoded):
            if result.get('error') is not None:
                raise result['error']
            audio = result['audio_path']
            samples, rate = sf.read(audio, dtype='float32', always_2d=True)
            seconds = len(samples) / rate
            if seconds < minimum or not np.isfinite(samples).all():
                result = self._invalid(*items[i], 'under minimum duration or nonfinite audio', audio)
            results[i] = result
        return results

    def _handle_failure(self, exc, request_id):
        if isinstance(exc, InvalidCompletion):
            return 0.0
        return super()._handle_failure(exc, request_id)


class AlignedSongEvalRM(InvalidOutputGuard, original.SongEvalRM):
    pass


class AlignedMuseCriticRM(InvalidOutputGuard, original.MuseCriticPCM24RM):
    pass


orms['songeval_rm'] = AlignedSongEvalRM
orms['musecritic_pcm24_rm'] = AlignedMuseCriticRM
