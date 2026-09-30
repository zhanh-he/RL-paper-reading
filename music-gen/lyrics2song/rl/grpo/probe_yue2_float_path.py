"""Locate YuE2 full-scale peaks before VAE clamping and PCM-24 export."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yue2
from peft import PeftModel

from yue2 import YuE2Pipeline
from yue2_songeval_longrun import HELDOUT, write_json


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def signal_stats(audio):
    waveform = np.asarray(audio, dtype=np.float32)
    if waveform.ndim == 1:
        waveform = waveform[:, None]
    absolute = np.abs(waveform)
    near = absolute >= 0.999
    longest = 0
    for channel in range(waveform.shape[1]):
        locations = np.flatnonzero(near[:, channel])
        if len(locations):
            boundaries = np.flatnonzero(np.diff(locations) > 1) + 1
            longest = max(longest, *(len(run) for run in np.split(locations, boundaries)))
    return {
        "peak": float(np.max(absolute)),
        "rms": float(np.sqrt(np.mean(waveform.astype(np.float64) ** 2))),
        "near_full_scale_count": int(np.count_nonzero(near)),
        "at_or_above_one_count": int(np.count_nonzero(absolute >= 1.0)),
        "above_one_count": int(np.count_nonzero(absolute > 1.0)),
        "longest_near_full_scale_run": longest,
    }


def decode_before_pipeline_clamp(pipe, latents):
    z = torch.as_tensor(latents, dtype=torch.float32)
    if z.ndim == 2 and z.shape[1] == 64:
        z = z.T.unsqueeze(0)
    if z.ndim != 3 or z.shape[0] != 1 or z.shape[1] != 64:
        raise ValueError("Unexpected YuE2 latent shape")
    model = pipe._vae.to(pipe.device)
    try:
        with torch.inference_mode():
            audio = model.decode_tiled(z, core_frames=pipe.vae_core_frames,
                                       halo_frames=16, output_device="cpu")
        if not torch.isfinite(audio).all():
            raise ValueError("VAE produced non-finite audio")
        return audio[0].float().T.contiguous().numpy()
    finally:
        model.to("cpu")
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--reference-flac", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heldout-index", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=600)
    args = parser.parse_args()
    if not 0 <= args.heldout_index < len(HELDOUT):
        parser.error("heldout-index is outside the original fixed evaluation set")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(90730)
    pipe = YuE2Pipeline.from_pretrained(args.model, vae=args.vae, device="cuda",
                                        backend="torch-eager", memory_budget_gib=32,
                                        progress=False)
    base = pipe._load_model().eval()
    PeftModel.from_pretrained(base, args.adapter, is_trainable=False).eval()
    index = args.heldout_index
    with torch.inference_mode():
        song = pipe(**{**HELDOUT[index], "seed": 5101 + index},
                    semantic_sampling={"max_tokens": args.max_tokens, "min_tokens": 200})
    unclamped = decode_before_pipeline_clamp(pipe, song.latents)
    raw = np.asarray(song.audio, dtype=np.float32)
    unclamped_path = args.output / "audio_preclamp_float.wav"
    raw_path = args.output / "audio_pipeline_float.wav"
    flac_path = args.output / "audio_pcm24.flac"
    sf.write(unclamped_path, unclamped, song.sample_rate, subtype="FLOAT")
    sf.write(raw_path, raw, song.sample_rate, subtype="FLOAT")
    sf.write(flac_path, raw, song.sample_rate, subtype="PCM_24")
    decoded, sample_rate = sf.read(flac_path, dtype="float32")
    reference, reference_rate = sf.read(args.reference_flac, dtype="float32")
    if sample_rate != song.sample_rate:
        raise AssertionError("Unexpected FLAC sample rate")
    comparable = reference_rate == sample_rate and reference.shape == decoded.shape
    exact_audio_match = comparable and bool(np.array_equal(reference, decoded))
    pipeline_match = unclamped.shape == raw.shape and bool(
        np.array_equal(np.clip(unclamped, -1, 1), raw)
    )
    receipt = {
        "status": "same_seed_vae_clamp_to_pcm24_export_probe",
        "optimizer_adapter": str(args.adapter),
        "yue2_version": yue2.__version__,
        "yue2_runtime_sha256": pipe.runtime_sha256,
        "heldout_index": index,
        "seed": 5101 + index,
        "sample_rate": song.sample_rate,
        "truncated": song.truncated,
        "vae_preclamp_float": signal_stats(unclamped),
        "pipeline_float": signal_stats(raw),
        "pcm24_flac": signal_stats(decoded),
        "pipeline_matches_clamped_vae": pipeline_match,
        "samples_changed_by_pipeline_clamp": int(np.count_nonzero(unclamped != raw))
        if pipeline_match else None,
        "reference_flac_sha256": sha256(args.reference_flac),
        "rerendered_flac_sha256": sha256(flac_path),
        "identical_to_reference_flac": sha256(args.reference_flac) == sha256(flac_path),
        "decoded_audio_identical_to_reference": exact_audio_match,
        "decoded_audio_max_abs_difference": (
            float(np.max(np.abs(reference - decoded))) if comparable else None
        ),
    }
    write_json(args.output / "receipt.json", receipt)
    print(receipt, flush=True)


if __name__ == "__main__":
    main()
