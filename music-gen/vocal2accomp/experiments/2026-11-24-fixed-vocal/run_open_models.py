"""Run a fixed-vocal candidate on an existing CUDA model install."""
import argparse
from datetime import datetime
from functools import lru_cache
from hashlib import sha256
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
from time import perf_counter
from zoneinfo import ZoneInfo

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
import torch

ANY_CHECKPOINT = {
    "id": "amphion/anyaccomp",
    "revision": "9aa9e62427337bf1df4caa3c4f3e6ad934522e711",
    "sha256": {
        "flow_matching": "e6802bd1123935a54e990cb8d3897a18190df6c53f73db021baa28c420721129",
        "vocoder": "7efd04c71c058cd00b4e9a91c761b31da745f878b7d7ee839e157104d3a7da",
        "vq": "9d7f48cefea30602b2148c057faf14ecad168184e8063c8377dd57f208dc65fc",
    },
}
LADA_CHECKPOINT = {
    "id": "sDuoluoluos/LaDA-Band/lada_band_lm1B_total3B.ckpt",
    "revision": "6d444caee85385677b0652ecb0b2b8220436dd37",
    "sha256": "050ee4e4c67fd7e06947f475d21af3a13e183344ce790e06a79076a148ef82c5",
}
STYLES = {
    "acoustic-folk": "Acoustic folk. Fingerpicked acoustic guitar, warm woody bass and understated brushed drums.",
    "sparse-synth-pop": "Sparse synth-pop. Restrained electric drum machine, rounded synth bass and airy sustained synth.",
    "half-time-rnb": "Half-time R&B. Soft kick, laid-back rimshot, warm electric piano and a smooth bass line.",
}
FOLLOW = (
    "Arrange around the supplied sung performance. Accompany its existing melody, "
    "timing, pickup, held notes and section changes. Keep the arrangement instrumental "
    "and leave space for the lead vocal."
)


def awst_time():
    return datetime.now(ZoneInfo("Australia/Perth")).isoformat()


def write_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def digest(path):
    digest_value = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest_value.update(chunk)
    return digest_value.hexdigest()


def audio_info(path):
    signal_data, sr = sf.read(path, dtype="float64", always_2d=True)
    if not signal_data.size or not np.isfinite(signal_data).all():
        raise ValueError("The waveform is empty or contains nonfinite values")
    return {
        "sha256": digest(path),
        "sample_rate": sr,
        "channels": signal_data.shape[1],
        "samples": signal_data.shape[0],
        "seconds": signal_data.shape[0] / sr,
        "rms": float(np.sqrt(np.mean(np.square(signal_data)))),
        "peak": float(np.max(np.abs(signal_data))),
    }


def checkpoint_info(repo, assets, is_lada):
    if is_lada:
        checkpoint = assets / "checkpoints" / "lada_band_lm1B_total3B.ckpt"
        actual = digest(checkpoint)
        expected = LADA_CHECKPOINT["sha256"]
        return dict(LADA_CHECKPOINT, actual_sha256=actual, matches_reference=actual == expected)
    checkpoints = {
        "flow_matching": repo / "pretrained" / "flow_matching" / "pytorch_model.bin",
        "vocoder": repo / "pretrained" / "vocoder" / "model.safetensors",
        "vq": repo / "pretrained" / "vq" / "pytorch_model.bin",
    }
    actual = {name: digest(path) for name, path in checkpoints.items()}
    return dict(ANY_CHECKPOINT, actual_sha256=actual, matches_reference=actual == ANY_CHECKPOINT["sha256"])


def resample(signal_data, source_rate, target_rate):
    divisor = math.gcd(source_rate, target_rate)
    return resample_poly(signal_data, target_rate // divisor, source_rate // divisor, axis=0)


def mix_fixed_vocal(input_fp, backing_fp, mix_fp):
    dry, dry_rate = sf.read(input_fp, dtype="float64", always_2d=True)
    backing, backing_rate = sf.read(backing_fp, dtype="float64", always_2d=True)
    backing = resample(backing, backing_rate, dry_rate)
    channels = max(dry.shape[1], backing.shape[1])
    mix = np.zeros((max(len(dry), len(backing)), channels))
    mix[: len(dry)] += 0.5 * dry
    mix[: len(backing)] += 0.5 * backing
    sf.write(mix_fp, mix, dry_rate, subtype="FLOAT")


def run_any(args, repo, receipt, dst):
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    import librosa
    from anyaccomp.inference_utils import Sing2SongInferencePipeline

    params = {"n_timesteps": 50, "cfg": 3.0, "precision": "CUDA BF16 autocast"}
    receipt["parameters"] = params
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    start = perf_counter()
    pipeline = Sing2SongInferencePipeline(
        "./pretrained/flow_matching",
        "./config/flow_matching.json",
        "./pretrained/vocoder",
        "./config/vocoder.json",
        device="cuda",
    )
    pipeline.sample_rate = 24000
    torch.cuda.synchronize()
    receipt["timings_seconds"]["model_load"] = perf_counter() - start

    start = perf_counter()
    vocal, _ = librosa.load(args.dry, sr=24000, mono=True)
    vocal_tensor = torch.tensor(vocal).unsqueeze(0).to("cuda")
    vocal_mel = pipeline.encode_vocal(vocal_tensor)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        mel = pipeline.model.reverse_diffusion(
            vocal_mel=vocal_mel, n_timesteps=params["n_timesteps"], cfg=params["cfg"]
        )
    wav = pipeline._generate_audio(mel.float()).squeeze().detach().cpu().numpy()
    torch.cuda.synchronize()
    receipt["timings_seconds"]["generation"] = perf_counter() - start
    receipt["model_input_samples"] = len(vocal)
    receipt["raw_output_samples"] = len(wav)
    sf.write(dst / "backing.wav", wav, 24000, subtype="FLOAT")


@lru_cache(maxsize=1)
def load_lada(repo):
    infer = repo / "codes" / "infer.py"
    spec = importlib.util.spec_from_file_location("lada_infer", infer)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_lada(args, repo, receipt, dst):
    child = dst / "upstream"
    assets = args.assets_root
    checkpoint = assets / "checkpoints" / "lada_band_lm1B_total3B.ckpt"
    infer_config = repo / "codes" / "lada_band" / "conf" / "infer_1B.yaml"
    os.environ["LADA_PRETRAINED_ROOT"] = str(assets / "pretrained")
    os.environ["LADA_CODE_ROOT"] = str(repo / "codes")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_CACHE"] = str(args.hf_cache or assets / "pretrained")
    os.chdir(repo / "codes")
    module = load_lada(repo)
    original_init = module.init_model

    def timed_init(*init_args, **init_kwargs):
        start = perf_counter()
        receipt["model_reused_in_process"] = module.model_module_list[0] is not None
        model = original_init(*init_args, **init_kwargs)
        torch.cuda.synchronize()
        receipt["timings_seconds"]["model_load"] = perf_counter() - start
        return model

    module.init_model = timed_init
    caption = f"{STYLES[args.style]} {FOLLOW}"
    receipt["caption"] = caption
    receipt["parameters"] = {
        "config": infer_config.name,
        "sampling_steps": args.lada_steps,
        "top_k": 100,
        "top_p": 0.9,
        "temperature": 1.0,
        "mask_temperature": 10.5,
        "remask_strategy": "low_confidence",
        "remask_schedule": "cosine",
        "decode_mode": "diffusion",
        "segment": [0, -1],
    }
    sys.argv = [
        str(repo / "codes" / "infer.py"),
        "--model_fp", str(checkpoint),
        "--config", str(infer_config),
        "--output_dir", str(child),
        "--seg_time", "0", "-1",
        "--vocal_fps", str(args.dry),
        "--texts", caption,
        "--sampling_steps", str(args.lada_steps),
        "--seed", str(args.seed),
        "--save_format", "wav",
        "--acc_only",
    ]
    try:
        module.batch_generate(module.parse_infer_args())
    finally:
        module.init_model = original_init
    generation_record, = sorted(child.glob("generation_results_*.json"))
    result, = json.loads(generation_record.read_text())
    receipt["sample_seed"] = result["sample_seed"]
    receipt["timings_seconds"]["generation"] = result["generation_time_seconds"]
    if result["status"] != "success":
        raise RuntimeError(result["error"])
    if result["reference_type"] != "text":
        raise RuntimeError("The upstream generation record did not report text conditioning")
    shutil.copyfile(result["generated_fp"], dst / "backing.wav")


def timed_out(_signal, _frame):
    global generation_timed_out
    generation_timed_out = True
    raise TimeoutError("Candidate exceeded its inference wall-clock limit")


def candidate_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=["anyaccomp", "lada-band"])
    parser.add_argument("dry", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--code-root", type=Path, required=True, help="Checkout of the selected upstream project")
    parser.add_argument("--assets-root", type=Path, help="LaDA-Band assets with pretrained/ and checkpoints/")
    parser.add_argument("--hf-cache", type=Path, help="Local Hugging Face cache for LaDA-Band")
    parser.add_argument("--style", choices=["unconditioned", *STYLES], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lada-steps", type=int, default=60)
    parser.add_argument("--timeout", type=int, default=600)
    return parser


def run_candidate(args):
    global generation_timed_out
    generation_timed_out = False
    args.dry = args.dry.resolve(strict=True)
    repo = args.code_root.resolve(strict=True)
    dst = args.out.resolve()
    is_lada = args.engine == "lada-band"
    if is_lada == (args.style == "unconditioned"):
        raise ValueError("LaDA-Band takes a style prompt. AnyAccomp takes dry vocal alone.")
    if is_lada and args.assets_root is None:
        raise ValueError("Provide the authorized LaDA-Band snapshot with --assets-root.")
    if args.assets_root:
        args.assets_root = args.assets_root.resolve(strict=True)
    if args.hf_cache:
        args.hf_cache = args.hf_cache.resolve(strict=True)
    experiment_repo = Path(__file__).resolve().parents[4]
    if dst.is_relative_to(experiment_repo) or dst.is_relative_to(repo):
        raise ValueError("Keep lossless audio and full receipts outside both Git checkouts.")
    os.umask(0o077)
    dst.mkdir(parents=True, exist_ok=False)
    start = perf_counter()
    receipt = {
        "schema_version": "fixed-vocal-inference-v1",
        "candidate_id": dst.name,
        "input": args.dry.name,
        "input_audio": audio_info(args.dry),
        "model": "LaDA-Band" if is_lada else "AnyAccomp",
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "style": args.style,
        "style_control": "text" if is_lada else "none",
        "model_output_kind": "isolated_backing",
        "seed_controllable": True,
        "seed": args.seed,
        "started_at_awst": awst_time(),
        "status": "running",
        "timings_seconds": {},
        "timeout_seconds": args.timeout,
        "outputs": {},
    }
    write_json(dst / "receipt.json", receipt)
    previous_handler = signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(args.timeout)
    try:
        hash_start = perf_counter()
        receipt["checkpoint"] = checkpoint_info(repo, args.assets_root, is_lada)
        receipt["timings_seconds"]["checkpoint_verification"] = perf_counter() - hash_start
        receipt["gpu"] = {
            "name": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        }
        torch.cuda.reset_peak_memory_stats()
        runner = run_lada if is_lada else run_any
        runner(args, repo, receipt, dst)
        postprocess_start = perf_counter()
        backing_fp = dst / "backing.wav"
        mix_fp = dst / "mix.wav"
        backing = audio_info(backing_fp)
        mix_fixed_vocal(args.dry, backing_fp, mix_fp)
        receipt["outputs"] = {"backing": backing, "mix": audio_info(mix_fp)}
        receipt["mix"] = {
            "provenance": "original dry and generated backing, unshifted fixed-gain sum",
            "dry_gain": 0.5,
            "backing_gain": 0.5,
        }
        receipt["duration_error_seconds"] = backing["seconds"] - receipt["input_audio"]["seconds"]
        receipt["timings_seconds"]["postprocess"] = perf_counter() - postprocess_start
        receipt["status"] = "success"
    except Exception as exc:
        receipt["status"] = "timeout" if isinstance(exc, TimeoutError) or generation_timed_out else "failed"
        receipt["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        if torch.cuda.is_initialized():
            receipt["gpu"]["caching_allocator_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
            receipt["gpu"]["caching_allocator_peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
        receipt["timings_seconds"]["total"] = perf_counter() - start
        receipt["ended_at_awst"] = awst_time()
        write_json(dst / "receipt.json", receipt)
    return receipt


generation_timed_out = False

if __name__ == "__main__":
    run_candidate(candidate_parser().parse_args())
