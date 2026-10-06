"""Register SongEval and canonical-FLAC MuseCritic rewards for Muse GRPO."""

import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import uuid
from pathlib import Path

import soundfile as sf
from swift.rewards import orms


base_path = Path(os.environ["MUSE_BASE_PLUGIN"])
spec = importlib.util.spec_from_file_location("muse_original_grpo_plugin", base_path)
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)


class SongEvalRM(base.MusecriticRM):
    def _score_audio(self, audio_path, request_id="", index=0, local_batch_size=1):
        scored = self._score_audios([audio_path], [request_id])[0]
        return scored["reward"], {"reward_scores": scored["reward_scores"], "infer_critic": ""}

    def _score_audios(self, audio_paths, request_ids):
        if not audio_paths:
            return []
        if len(audio_paths) != len(request_ids):
            raise ValueError("SongEval audio and request IDs are not aligned")
        repo = Path(os.environ["SONGEVAL_REPO"])
        python = os.environ["SONGEVAL_PY"]
        batch = Path(os.environ["MUSE_SCORING_DIR"]) / uuid.uuid4().hex
        batch.mkdir(parents=True, exist_ok=False)
        paths = batch / "inputs.txt"
        paths.write_text("".join(f"{Path(path).resolve()}\n" for path in audio_paths))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("MUSE_SERVICE_GPU", "1"))
        subprocess.run([python, "eval.py", "-i", str(paths), "-o", str(batch / "scores")],
                       cwd=repo, env=env, check=True)
        scored = json.loads((batch / "scores" / "result.json").read_text())
        results = []
        with (batch / "bindings.jsonl").open("w") as receipt:
            for path, request_id in zip(audio_paths, request_ids):
                source = Path(path)
                dimensions = scored[source.stem]
                reward = sum(float(value) for value in dimensions.values()) / len(dimensions)
                if len(dimensions) != 5 or not math.isfinite(reward):
                    raise ValueError(f"Nonfinite or incomplete SongEval score for {request_id}")
                receipt.write(json.dumps({"request_id": request_id, "audio_path": str(source),
                                          "audio_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                                          "dimensions": dimensions, "reward": reward}) + "\n")
                results.append({"reward": reward, "reward_scores": dimensions,
                                "infer_critic": "", "error": None})
        return results


class MuseCriticPCM24RM(base.MusecriticRM):
    def _decode_completion_to_wav(self, completion, request_id, index, local_batch_size=1):
        wav = Path(super()._decode_completion_to_wav(completion, request_id, index,
                                                     local_batch_size))
        samples, sample_rate = sf.read(wav, dtype="float32")
        canonical = wav.with_suffix(".flac")
        sf.write(canonical, samples, sample_rate, subtype="PCM_24")
        return str(canonical)


orms["songeval_rm"] = SongEvalRM
orms["musecritic_pcm24_rm"] = MuseCriticPCM24RM
