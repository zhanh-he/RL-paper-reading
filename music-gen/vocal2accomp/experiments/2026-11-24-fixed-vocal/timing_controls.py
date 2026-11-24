#!/usr/bin/env python3
"""Generate and analyze known timing controls for the accompaniment detector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from evaluate_timing import Detector, analyze, pair_events, public_detector, sha256

RATE = 24000
BEAT_S = 60 / 110
FIRST_BEAT_S = 0.91
BARS = 10
DURATION_S = round((FIRST_BEAT_S + BARS * 4 * BEAT_S + 0.7) * RATE) / RATE
LATE_S = 0.42
MUTED_BARS = (4, 5)


def make_control(delay_s=0.0, mute_bars=()):
    audio = np.zeros(round(DURATION_S * RATE))
    rng = np.random.default_rng(7217)
    time = np.arange(round(0.32 * RATE)) / RATE
    kick = np.sin(2 * np.pi * (45 * time + 25 * (1 - np.exp(-60 * time)) / 60)) * np.exp(-18 * time)

    def add(at_s, pulse):
        index = round(at_s * RATE)
        pulse = pulse[:len(audio) - index]
        audio[index:index + len(pulse)] += pulse

    for i in range(BARS * 4):
        snare = np.diff(rng.normal(size=len(time) + 1)) * np.exp(-30 * time)
        hat = np.diff(rng.normal(size=len(time) + 1)) * np.exp(-75 * time)
        bar_index = i // 4
        if bar_index in mute_bars:
            continue
        at_s = FIRST_BEAT_S + i * BEAT_S + delay_s
        root = (110, 130.81, 98, 146.83)[bar_index % 4]
        bass = (np.sin(2 * np.pi * root * time) + np.sin(4 * np.pi * root * time) * 0.45)
        bass *= np.exp(-9 * time) * np.minimum(1, 220 * time)
        chord = sum(np.sin(2 * np.pi * root * q * 2 * time) for q in (1, 1.2, 1.5))
        chord *= np.exp(-18 * time) * np.minimum(1, 600 * time)
        if i % 4 in (0, 2):
            add(at_s, kick * (0.52 if i % 4 == 0 else 0.25))
            add(at_s, bass * 0.11)
        else:
            add(at_s, snare * 0.045)
        add(at_s, chord * (0.09 if i % 4 == 0 else 0.035))
        add(at_s + BEAT_S / 2, hat * 0.02)
    return audio


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def validate(results):
    aligned = results['aligned']
    late = results['late']
    drops = results['two_bars_muted']
    delayed_entry = results['first_bar_muted']
    silence = results['silence']
    short = results['cropped']
    return {
        'aligned_detects_all_raw': aligned['raw']['match_count'] == BARS,
        'aligned_estimates_near_zero_shift': aligned['global_shift']['estimate_s'] is not None and abs(aligned['global_shift']['estimate_s']) < 0.05,
        'late_raw_reports_misses': late['raw']['match_count'] == 0,
        'late_estimates_known_shift': late['global_shift']['estimate_s'] is not None and abs(late['global_shift']['estimate_s'] - LATE_S) < 0.05,
        'late_correction_recovers_pairs': late['corrected'] is not None and late['corrected']['match_count'] == BARS,
        'first_bar_absent_and_entry_flagged': delayed_entry['entry_flag'] == 'late_possible' and any(
            row['bar'] == 1 and row['reason'] == 'audio_blank_at_reference'
            for row in delayed_entry['raw']['unmatched_reference']
        ),
        'silence_keeps_no_model_events': silence['detector_status'] == 'no_signal' and not silence['detections']['downbeats'],
        'silence_records_no_shift_evidence': silence['corrected'] is None and len(silence['raw']['unmatched_reference']) == BARS,
        'cropped_detects_shortfall': short['duration_flag'] == 'possibly_cropped' and any(
            row['reason'] == 'beyond_output_end' for row in short['raw']['unmatched_reference']
        ),
        'missing_bars_are_independently_flagged_blank': {row['bar'] for row in drops['raw']['unmatched_reference'] if row['reason'] == 'audio_blank_at_reference'} == {bar + 1 for bar in MUTED_BARS},
        'one_to_one_does_not_reuse_detection': pair_events([1.0, 1.2], [1.08], 0.15)[0] == [(0, 0)],
        'one_to_one_maximizes_coverage': pair_events([1.0, 1.16], [0.9, 1.06], 0.15)[0] == [(0, 0), (1, 1)],
        'one_to_one_leaves_extra_detection': pair_events([1.0], [0.91, 1.03, 2.0], 0.15)[0] == [(0, 1)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--source-commit', help='checkout revision for the internal receipt')
    parser.add_argument('--public-summary', type=Path)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    reference = args.work_dir / 'reference.json'
    write_json(reference, {
        'source': 'Original deterministic percussion and chord PCM control',
        'duration_s': DURATION_S,
        'bars': [{
            'bar': i + 1,
            'time_s': FIRST_BEAT_S + i * 4 * BEAT_S,
            'section': 'A' if i < 5 else 'B',
        } for i in range(BARS)],
        'uncertain_intervals': [],
    })
    aligned = make_control()
    signals = {
        'aligned': aligned,
        'late': make_control(delay_s=LATE_S),
        'first_bar_muted': make_control(mute_bars=(0,)),
        'two_bars_muted': make_control(mute_bars=MUTED_BARS),
        'silence': np.zeros_like(aligned),
        'cropped': aligned[:round((FIRST_BEAT_S + 4 * 8 * BEAT_S - 0.4) * RATE)],
    }
    detector = Detector(bundle, bundle / 'weights/beatthis_mirex.ckpt', bundle / 'config/casm-no-smc.json',
                        args.source_commit)
    results = {}
    for name, audio in signals.items():
        path = args.work_dir / f'{name}.wav'
        sf.write(path, audio, RATE, subtype='PCM_24')
        result = analyze(detector, path, reference, 'control', 0.15, 0.75)
        results[name] = result
        write_json(args.work_dir / f'{name}.json', result)
    validation = validate(results)
    summary = {
        'schema': 'accompaniment-timing-control/v1',
        'reference_sha256': sha256(reference),
        'procedure': 'Downbeats inferred from each rendered PCM control independently of the control reference',
        'known_lag_s': LATE_S,
        'known_muted_bar_ids': [bar + 1 for bar in MUTED_BARS],
        'detector': detector.provenance,
        'results': {name: {
            'audio_sha256': result['audio_sha256'],
            'status': result['detector_status'],
            'analysis_seconds': result['analysis_seconds'],
            'entry_flag': result['entry_flag'],
            'first_downbeat_raw_lag_s': result['first_downbeat_raw_lag_s'],
            'duration_flag': result['duration_flag'],
            'raw_match_count': result['raw']['match_count'],
            'raw_extra_count': len(result['raw']['extra_detected']),
            'raw_unmatched_reference': result['raw']['unmatched_reference'],
            'global_shift_s': result['global_shift']['estimate_s'],
            'corrected_match_count': result['corrected']['match_count'] if result['corrected'] else None,
            'corrected_median_absolute_residual_s': result['corrected']['median_absolute_residual_s'] if result['corrected'] else None,
            'suppressed_downbeats': [event for event in result['detections']['downbeats'] if event['suppression_reason']],
        } for name, result in results.items()},
        'checks': validation,
    }
    write_json(args.work_dir / 'summary.json', summary)
    if args.public_summary:
        args.public_summary.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.public_summary, {**summary, 'detector': public_detector(detector.provenance)})
    failed = [name for name, passed in validation.items() if not passed]
    if failed:
        raise SystemExit(f'Failed controls: {", ".join(failed)}')


if __name__ == '__main__':
    main()
