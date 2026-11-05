"""Decompose existing auditory/roughness scores without generating audio.

Requires numpy. All input/output locations are explicit CLI arguments. Inputs
can be split worker results or the combined public naturalness score files.
Only explicitly constructed numerical/provenance fields enter the output.
"""
import argparse
import collections
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

REGIONS = (
    "low_centers_lt1500", "mid_centers1500_3000",
    "high_centers3000_8000", "upper_centers_gt8000",
)
GROUPS = ("suno", "diff", "bloom", "muse")
# Recursive publication whitelist: unexpected keys fail instead of copying
# arbitrary input metadata into the public result.
PUBLIC_KEYS = set(GROUPS + REGIONS) | set("""
schema date scope human_labels_count interpretation_limits input_sha256
suno_auditory suno_roughness controls_auditory controls_roughness
aggregates muse_fixed_codes rows roughness_temporal_rows checks
id audio_sha256 global_M reconstructed_M maximum_scalar_reconstruction_error
dominant_channel_center_hz reported_peak_modulation_hz regions
regional_weighted_modulation aggregation_weight_share contribution_to_global_M
share_of_global_M channel_count roughness_mean roughness_median top_frames
total_frames top_about10pct_frame_share_of_sum frames_gt1asper clips
low_center_contribution_share high_center_contribution_share
clips_low_contribution_above_half clips_high_contribution_above_half
peak_modulation_rate_counts top_about10pct_roughness_frame_share
clips_top_about10pct_frames_contribute_over_half minimum median maximum
roughness_temporal total_real_excerpts maximum_global_M_reconstruction_error
all_audio_hashes_match_existing_scores
""".split())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_samples(path):
    data = json.loads(Path(path).read_text())
    if "status" in data and data["status"] != "complete":
        raise ValueError("Input score file is incomplete")
    return data["samples"]


def safe_identity(row):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", row["id"]):
        raise ValueError("Sample IDs must be publication-safe identifiers")
    if not re.fullmatch(r"[0-9a-f]{64}", row["audio_sha256"]):
        raise ValueError("Invalid audio SHA256")
    return {"id": row["id"], "audio_sha256": row["audio_sha256"]}


def summary(values):
    x = np.asarray(values, dtype=float)
    return dict(minimum=float(x.min()), median=float(np.median(x)),
                maximum=float(x.max()))


def decompose(row):
    g = row["auditory"]
    m = g["maps"]
    f = np.asarray(m["center_hz"])
    depth = np.asarray(m["modulation_depth"])
    energy = np.asarray(m["band_energy_relative"])
    weight = energy ** 2 * (energy > 1e-3)
    total = weight.sum()
    contribution = (depth * weight).sum(1) / total
    masks = (f < 1500, (f >= 1500) & (f < 3000),
             (f >= 3000) & (f <= 8000), f > 8000)
    regions = {}
    for name, mask in zip(REGIONS, masks):
        denominator = weight[mask].sum()
        value = float((depth[mask] * weight[mask]).sum() / denominator) if denominator else None
        part = float(contribution[mask].sum())
        regions[name] = dict(
            regional_weighted_modulation=value,
            aggregation_weight_share=float(denominator / total),
            contribution_to_global_M=part,
            share_of_global_M=part / float(contribution.sum()),
            channel_count=int(mask.sum()),
        )
    return dict(**safe_identity(row), global_M=g["modulation_depth_20_300"],
                reconstructed_M=float(contribution.sum()),
                maximum_scalar_reconstruction_error=abs(float(contribution.sum()) - g["modulation_depth_20_300"]),
                dominant_channel_center_hz=float(f[contribution.argmax()]),
                reported_peak_modulation_hz=g["fast_modulation_peak_hz"],
                regions=regions)


def temporal(row):
    if "roughness70" in row:
        roughness = row["roughness70"]
    elif "roughness" in row:
        roughness = row["roughness"]["levels"]["70"]
    else:
        roughness = row
    x = np.asarray(roughness["time_asper"])
    n = int(math.ceil(len(x) * .1))
    share = float(np.sort(x)[-n:].sum() / x.sum())
    return dict(**safe_identity(row), roughness_mean=float(x.mean()),
                roughness_median=float(np.median(x)), top_frames=n,
                total_frames=len(x), top_about10pct_frame_share_of_sum=share,
                frames_gt1asper=int((x > 1).sum()))


def control_group(row):
    group = row["id"].split("_", 1)[0]
    if group not in GROUPS[1:]:
        raise ValueError("Control sample ID has an unsupported model prefix")
    return group


def check_public_tree(value, parent=None):
    if isinstance(value, dict):
        for key, item in value.items():
            # Peak rates are numerical JSON object keys, not arbitrary metadata.
            if key not in PUBLIC_KEYS and not (parent == "peak_modulation_rate_counts" and re.fullmatch(r"[0-9.]+", key)):
                raise ValueError("Unexpected publication key: " + key)
            check_public_tree(item, key)
    elif isinstance(value, list):
        for item in value:
            check_public_tree(item, parent)
    elif isinstance(value, str):
        if value.startswith("/") or re.search(r"\w+://|\S+@\S+", value):
            raise ValueError("Unexpected private-location text in result")


def build(args):
    rows = {group: [] for group in GROUPS}
    temps = {group: [] for group in GROUPS}
    rows["suno"] = [decompose(row) for row in read_samples(args.suno_auditory)]
    temps["suno"] = [temporal(row) for row in read_samples(args.suno_roughness)]
    for path in args.controls_auditory:
        for row in read_samples(path):
            rows[control_group(row)].append(decompose(row))
    for path in args.controls_roughness:
        for row in read_samples(path):
            temps[control_group(row)].append(temporal(row))
    for group in GROUPS:
        if not rows[group] or not temps[group]:
            raise ValueError("All four existing source groups are required")
        ids = [row["id"] for row in rows[group]]
        rough_ids = [row["id"] for row in temps[group]]
        if len(set(ids)) != len(ids) or len(set(rough_ids)) != len(rough_ids):
            raise ValueError("Duplicate sample IDs")
        if {r["id"]: r["audio_sha256"] for r in rows[group]} != {r["id"]: r["audio_sha256"] for r in temps[group]}:
            raise ValueError("Audio hashes differ between descriptor inputs")
    aggregates = {}
    for group in GROUPS:
        rr = rows[group]
        low = [r["regions"][REGIONS[0]]["share_of_global_M"] for r in rr]
        high = [r["regions"][REGIONS[2]]["share_of_global_M"] for r in rr]
        t = [r["top_about10pct_frame_share_of_sum"] for r in temps[group]]
        aggregates[group] = dict(
            clips=len(rr), low_center_contribution_share=summary(low),
            high_center_contribution_share=summary(high),
            clips_low_contribution_above_half=sum(x > .5 for x in low),
            clips_high_contribution_above_half=sum(x > .5 for x in high),
            peak_modulation_rate_counts=dict(collections.Counter(str(r["reported_peak_modulation_hz"]) for r in rr)),
            top_about10pct_roughness_frame_share=summary(t),
            clips_top_about10pct_frames_contribute_over_half=sum(x > .5 for x in t),
        )
    muse = [r for r in rows["muse"] if r["id"].startswith("muse_synth_s37_base")]
    tempby = {r["id"]: r for r in temps["muse"]}
    for row in muse:
        row["roughness_temporal"] = tempby[row["id"]]
    result = dict(
        schema="descriptor-contribution-audit-v2", date="2026-11-05",
        scope="Arithmetic decomposition of previously scored real Suno/generated audio; no new audio, simulation, training or human labels.",
        human_labels_count=0,
        interpretation_limits=[
            "Groups contain dependent windows/renders, not independent repetitions; no confidence intervals or detection accuracy.",
            "Channel-center groups are not brick-wall bands. Aggregation weights are squared envelope means, not calibrated acoustic power or loudness.",
            "Regional modulation is an exploratory descriptor; a rise or fall is not a human artifact verdict.",
            "Top-frame concentration uses overlapping200ms DW windows at100ms hop; it is not exact event duration or a transient classification.",
            "Reported24Hz peak can be influenced by the20Hz lower bound and5bin smoothing; it is not proof of24Hz periodic artifacts.",
            "Evidence from public implementations cannot be extrapolated to other implementations or private internal systems.",
        ],
        input_sha256=dict(suno_auditory=sha(args.suno_auditory), suno_roughness=sha(args.suno_roughness),
                          controls_auditory=[sha(p) for p in args.controls_auditory],
                          controls_roughness=[sha(p) for p in args.controls_roughness]),
        aggregates=aggregates, muse_fixed_codes=muse, rows=rows,
        roughness_temporal_rows=temps,
        checks=dict(total_real_excerpts=sum(len(v) for v in rows.values()),
                    maximum_global_M_reconstruction_error=max(r["maximum_scalar_reconstruction_error"] for rr in rows.values() for r in rr),
                    all_audio_hashes_match_existing_scores=True),
    )
    check_public_tree(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suno-auditory", required=True, help="Suno auditory JSON")
    parser.add_argument("--suno-roughness", required=True, help="Suno DW70 JSON")
    parser.add_argument("--controls-auditory", required=True, nargs="+", help="One combined or several split controls auditory JSON files")
    parser.add_argument("--controls-roughness", required=True, nargs="+", help="Matching combined or split controls DW70 JSON files")
    parser.add_argument("--output", required=True, help="Destination JSON")
    args = parser.parse_args()
    result = build(args)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["checks"], indent=2))


if __name__ == "__main__":
    main()
