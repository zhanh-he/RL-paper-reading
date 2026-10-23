"""Canonical FLAC rewards on the existing Muse multi-turn GRPO integration."""
import importlib.util
import json
import os
import sys
from pathlib import Path
import soundfile as sf
import numpy as np
import torch
from swift.rewards import orms
from swift.rollout.multi_turn import multi_turns
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rewards import score_remote

spec = importlib.util.spec_from_file_location('regime_aligned', os.environ['MUSE_ALIGNED_PLUGIN'])
aligned = importlib.util.module_from_spec(spec); sys.modules[spec.name] = aligned; spec.loader.exec_module(aligned)


class RegimeScheduler(aligned.original.base.MuseMultiTurnScheduler):
    MAX_TOKENS_PER_TURN = 512


multi_turns['regime_scheduler'] = RegimeScheduler


class RegimeRM(aligned.InvalidOutputGuard, aligned.original.base.MusecriticRM):
    def _decode_completion_to_wav(self, completion, request_id, index, local_batch_size=1):
        torch.cuda.empty_cache()
        source = Path(super()._decode_completion_to_wav(completion, request_id, index, local_batch_size))
        samples, sr = sf.read(source, dtype='float32')
        if not np.isfinite(samples).all():
            return str(source)  # Preserve nonfinite values for the inherited invalid-output guard.
        destination = source.with_suffix('.flac')
        sf.write(destination, samples, sr, subtype='PCM_24')
        return str(destination)

    def _score_audios(self, paths, request_ids):
        rows = score_remote(os.environ['REGIME_SCORER'], paths, os.environ['REGIME_REWARD'])
        if os.environ.get('REGIME_BINDINGS'):
            with Path(os.environ['REGIME_BINDINGS']).open('a') as target:
                for request_id, row in zip(request_ids, rows):
                    target.write(json.dumps(dict(request_id=request_id, **row), allow_nan=False)+'\n')
        results = []
        for row in rows:
            components = {k:v for k,v in row['songbench'].items() if v is not None}
            components['Technical_invalid'] = float(row.get('technical_invalid',False))
            if row['richness']['raw'] is not None:
                components['SP_Richness'] = row['richness']['raw']
            if row['reward'] != 'mean7':
                components['Shared_penalty'] = row['guard']['penalty']
            results.append({'reward': row['mean'], 'reward_scores': components,
                            'infer_critic': '', 'error': None})
        return results

    def _score_audio(self, path, request_id='', index=0, local_batch_size=1):
        result = self._score_audios([path], [request_id])[0]
        return result['reward'], result


orms['regime_rm'] = RegimeRM
