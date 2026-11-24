#!/usr/bin/env python3
"""Compare audio-derived downbeat proposals with an independent reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

VERSION = '1.0.0'
FPS = 50
RMS_FRAME_S = 0.02
SIGNAL_WINDOW_S = 0.10
MIN_BLANK_S = 0.35
ABSOLUTE_GATE_DBFS = -55.0
RELATIVE_GATE_DB = -45.0
LOW_CONFIDENCE = 0.25


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def span_contains(span, time_s):
    return span['start_s'] <= time_s < span['end_s']


def detect_signal(path):
    import numpy as np
    import soundfile as sf

    audio, rate = sf.read(path, dtype='float32', always_2d=True)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError('audio must contain finite samples')
    frame_samples = max(1, round(rate * RMS_FRAME_S))
    frames = (len(audio) + frame_samples - 1) // frame_samples
    frame_s = frame_samples / rate
    duration_s = len(audio) / rate
    energy = np.square(np.pad(audio, ((0, frames * frame_samples - len(audio)), (0, 0))))
    powers = energy.reshape(frames, frame_samples, audio.shape[1]).mean(axis=(1, 2))
    window_frames = min(frames, round(SIGNAL_WINDOW_S / frame_s))
    smoothed = np.convolve(powers, np.full(window_frames, 1.0 / window_frames), mode='same')
    signal_dbfs = 10 * np.log10(np.maximum(smoothed, 1e-20))
    threshold = float(max(ABSOLUTE_GATE_DBFS, signal_dbfs.max() + RELATIVE_GATE_DB))
    active = np.flatnonzero(signal_dbfs >= threshold)
    transitions = np.flatnonzero(np.diff(np.r_[0, signal_dbfs < threshold, 0]))
    blanks = []
    for start, end in zip(transitions[::2], transitions[1::2]):
        first, last = start * frame_s, min(end * frame_s, duration_s)
        if last - first >= MIN_BLANK_S or not len(active):
            blanks.append({'start_s': first, 'end_s': last, 'reason': 'audio_below_rms_gate'})
    return {
        'duration_s': duration_s,
        'sample_rate_hz': rate,
        'channels': audio.shape[1],
        'peak_absolute_sample': float(np.abs(audio).max()),
        'out_of_range_sample_count': int(np.count_nonzero(np.abs(audio) > 1.0)),
        'gate_dbfs': threshold,
        'gate_envelope_frame_s': frame_s,
        'gate_envelope_window_s': SIGNAL_WINDOW_S,
        'minimum_blank_s': MIN_BLANK_S,
        'first_signal_s': float(active[0] * frame_s) if len(active) else None,
        'last_signal_s': min(float((active[-1] + 1) * frame_s), duration_s) if len(active) else None,
        'blank_intervals': blanks,
    }


def activation_at(logits, time_s):
    index = min(len(logits) - 1, max(0, round(time_s * FPS)))
    logit = float(logits[index])
    return 1.0 / (1.0 + math.exp(-max(-80, min(80, logit))))


class Detector:
    def __init__(self, bundle, checkpoint, config, source_commit=None):
        import torch
        import torchaudio

        sys.path[:0] = [str(bundle), str(bundle / 'third_party/beat_this'), str(bundle / 'vendor/casm/src')]
        from mirex_pipeline.beatthis_backend import BeatThisBackend
        from mirex_pipeline.decoders import CASMEventDecoder

        torch.set_num_threads(4)
        self.backend = BeatThisBackend(checkpoint=checkpoint, device='cpu', fps=FPS)
        self.decoder = CASMEventDecoder(config_path=config)
        self.provenance = {
            'backbone': 'Beat This (CASM MIREX 2026 split checkpoint)',
            'checkpoint': checkpoint.name,
            'checkpoint_sha256': sha256(checkpoint),
            'postprocessor': 'CASM no-SMC',
            'config': config.name,
            'config_sha256': sha256(config),
            'bundle': bundle.name,
            'source_checkout_commit': source_commit,
            'mirex_backend_source_sha256': sha256(bundle / 'mirex_pipeline/beatthis_backend.py'),
            'mirex_decoder_source_sha256': sha256(bundle / 'mirex_pipeline/decoders.py'),
            'activation_hz': FPS,
            'torch': str(torch.__version__),
            'torchaudio': str(torchaudio.__version__),
            'device': 'cpu',
            'confidence': 'Sigmoid of the raw activation logit at the decoded event frame, uncalibrated',
            'low_confidence_below': LOW_CONFIDENCE,
        }

    def run(self, path, signal):
        activations = self.backend.predict(path)
        beats, downbeats = self.decoder.decode(activations)
        return {
            'beats': find_events(beats, activations.beat_logits, signal),
            'downbeats': find_events(downbeats, activations.downbeat_logits, signal),
        }


def find_events(times, logits, signal):
    events = []
    for seconds in times:
        time_s = float(seconds)
        confidence = activation_at(logits, time_s)
        if not 0 <= time_s < signal['duration_s']:
            kind = 'outside_audio'
        elif any(span_contains(blank, time_s) for blank in signal['blank_intervals']):
            kind = 'audio_below_rms_gate'
        else:
            kind = None
        events.append({
            'time_s': time_s,
            'confidence': confidence,
            'low_confidence': confidence < LOW_CONFIDENCE,
            'suppression_reason': kind,
        })
    return events


def pair_events(reference_s, detected_s, tolerance_s, shift_s=0.0):
    """Maximize monotonic one-to-one coverage, then minimize total absolute error."""
    n_ref, n_det = len(reference_s), len(detected_s)
    scores = [[(0, 0.0)] * (n_det + 1) for _ in range(n_ref + 1)]
    moves = [[None] * (n_det + 1) for _ in range(n_ref + 1)]
    for i in range(1, n_ref + 1):
        for j in range(1, n_det + 1):
            options = [(scores[i - 1][j], 'skip_ref'), (scores[i][j - 1], 'skip_det')]
            distance = abs(detected_s[j - 1] - reference_s[i - 1] - shift_s)
            if distance <= tolerance_s + 1e-9:
                count, cost = scores[i - 1][j - 1]
                options.append(((count + 1, cost - distance), 'match'))
            scores[i][j], moves[i][j] = max(options)
    i, j = n_ref, n_det
    pairs = []
    while i and j:
        move = moves[i][j]
        if move == 'match':
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif move == 'skip_ref':
            i -= 1
        else:
            j -= 1
    return list(reversed(pairs)), -scores[n_ref][n_det][1]


def estimate_shift(reference_s, detected_s, tolerance_s, max_shift_s):
    support_min = max(3, math.ceil(0.45 * len(reference_s)))
    hypotheses = {0.0}
    hypotheses.update(
        round(detected - reference, 6)
        for reference in reference_s for detected in detected_s
        if abs(detected - reference) <= max_shift_s
    )
    best_shift = None
    best_key = None
    for hypothesis in sorted(hypotheses):
        pairs, _ = pair_events(reference_s, detected_s, tolerance_s, hypothesis)
        if len(pairs) < support_min:
            continue
        center_s = median(detected_s[j] - reference_s[i] for i, j in pairs)
        if abs(center_s) > max_shift_s:
            continue
        corrected, error = pair_events(reference_s, detected_s, tolerance_s, center_s)
        key = (len(corrected), -round(error, 8), -abs(center_s))
        if len(corrected) >= support_min and (best_key is None or key > best_key):
            best_key, best_shift = key, center_s
    return {
        'estimate_s': best_shift,
        'matched_support': best_key[0] if best_key is not None else 0,
        'min_support': support_min,
        'eligible_reference_count': len(reference_s),
        'eligible_detection_count': len(detected_s),
        'search_limit_s': max_shift_s,
        'status': 'estimated' if best_key is not None else 'insufficient_evidence',
    }


def compare(reference, detected, tolerance_s, shift_s, signal, uncertain):
    reference_s = [bar['time_s'] for bar in reference]
    detected_s = [event['time_s'] for event in detected]
    pairs, _ = pair_events(reference_s, detected_s, tolerance_s, shift_s)
    by_reference = dict(pairs)
    used_detections = set(by_reference.values())
    matches, missing, extra = [], [], []
    for i, bar in enumerate(reference):
        uncertainty = [span['reason'] for span in uncertain if span_contains(span, bar['time_s'])]
        if i not in by_reference:
            audio_time_s = bar['time_s'] + shift_s
            if audio_time_s >= signal['duration_s']:
                reason = 'beyond_output_end'
            elif any(span_contains(blank, audio_time_s) for blank in signal['blank_intervals']):
                reason = 'audio_blank_at_reference'
            else:
                reason = 'no_paired_downbeat_within_tolerance'
            missing.append({**bar, 'reason': reason, 'reference_uncertainty': uncertainty})
            continue
        event = detected[by_reference[i]]
        lag = event['time_s'] - bar['time_s']
        residual = lag - shift_s
        direction = 'late' if residual > 0.05 else 'early' if residual < -0.05 else 'within_50ms'
        matches.append({
            **bar, 'detected_s': event['time_s'], 'raw_lag_s': lag, 'residual_s': residual,
            'confidence': event['confidence'], 'low_confidence': event['low_confidence'],
            'reference_uncertainty': uncertainty, 'timing_flag': direction,
        })
    for j, event in enumerate(detected):
        if j not in used_detections:
            nearest = min(reference, key=lambda bar: abs(event['time_s'] - bar['time_s'] - shift_s))
            extra.append({
                **event, 'nearest_bar': nearest['bar'],
                'raw_lag_s_to_nearest_bar': event['time_s'] - nearest['time_s'],
                'residual_s_to_nearest_bar': event['time_s'] - nearest['time_s'] - shift_s,
                'reason': 'downbeat_without_unique_reference_match',
            })
    lag = [match['raw_lag_s'] for match in matches]
    residuals = [abs(match['residual_s']) for match in matches]
    return {
        'shift_applied_to_detection_s': -shift_s,
        'match_count': len(matches),
        'reference_count': len(reference),
        'eligible_detection_count': len(detected),
        'median_raw_lag_s': median(lag) if lag else None,
        'median_absolute_residual_s': median(residuals) if residuals else None,
        'matched': matches,
        'unmatched_reference': missing,
        'extra_detected': extra,
    }


def parse_reference(path):
    reference = json.loads(path.read_text())
    duration = float(reference['duration_s'])
    bars = reference['bars']
    times = [float(bar['time_s']) for bar in bars]
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('reference duration must be positive and finite')
    if not times or any(not 0 <= value < duration for value in times) or any(
        first >= second for first, second in zip(times, times[1:])
    ):
        raise ValueError('reference must have increasing, in-range downbeats')
    if len({bar['bar'] for bar in bars}) != len(bars):
        raise ValueError('reference bar IDs must be unique')
    for bar, time_s in zip(bars, times):
        bar['time_s'] = time_s
    uncertain = reference.get('uncertain_intervals', [])
    if any(not 0 <= span['start_s'] < span['end_s'] <= duration for span in uncertain):
        raise ValueError('uncertain intervals must be nonempty and in range')
    return duration, bars, uncertain, reference


def analyze(detector, path, reference_path, kind, tolerance_s, max_shift_s):
    started = perf_counter()
    reference_duration_s, bars, uncertainty, reference = parse_reference(reference_path)
    signal = detect_signal(path)
    if signal['first_signal_s'] is None:
        detections = {'beats': [], 'downbeats': []}
        detector_status = 'no_signal'
    else:
        detections = detector.run(path, signal)
        detector_status = 'events_proposed'
    eligible = [event for event in detections['downbeats'] if event['suppression_reason'] is None]
    if not eligible and detector_status != 'no_signal':
        detector_status = 'no_eligible_downbeat'
    supported = [event['time_s'] for event in eligible if not event['low_confidence']]
    certain_ref = [bar['time_s'] for bar in bars if not any(
        span_contains(span, bar['time_s']) for span in uncertainty
    )]
    shift = estimate_shift(certain_ref, supported, tolerance_s, max_shift_s)
    raw = compare(bars, eligible, tolerance_s, 0.0, signal, uncertainty)
    corrected = None
    if shift['estimate_s'] is not None:
        corrected = compare(bars, eligible, tolerance_s, shift['estimate_s'], signal, uncertainty)
    summary_view = corrected if corrected is not None else raw
    section_changes = [bar['bar'] for prev, bar in zip(bars, bars[1:]) if prev.get('section') != bar.get('section')]
    section_boundaries = []
    for section in reference.get('sections', []):
        start_s = section['start_s']
        closest = min(eligible, key=lambda event: abs(event['time_s'] - start_s), default=None)
        section_boundaries.append({
            'name': section['name'],
            'reference_start_s': start_s,
            'nearest_downbeat_s': closest['time_s'] if closest else None,
            'raw_lag_s': closest['time_s'] - start_s if closest else None,
            'confidence': closest['confidence'] if closest else None,
            'reference_uncertainty': [span['reason'] for span in uncertainty if span_contains(span, start_s)],
        })
    first_downbeat_s = eligible[0]['time_s'] if eligible else None
    first_lag = first_downbeat_s - bars[0]['time_s'] if first_downbeat_s is not None else None
    output_delta_s = signal['duration_s'] - reference_duration_s
    max_duration_difference_s = 0.25
    return {
        'schema': 'accompaniment-timing/v1',
        'version': VERSION,
        'audio_kind': kind,
        'audio_sha256': sha256(path),
        'reference_sha256': sha256(reference_path),
        'reference_type': reference.get('reference_type', reference.get('source')),
        'reference_basis': reference.get('reference_basis'),
        'detector': detector.provenance,
        'detector_status': detector_status,
        'pairing_tolerance_s': tolerance_s,
        'duration_delta_s': output_delta_s,
        'duration_flag': 'possibly_cropped' if output_delta_s < -max_duration_difference_s else
                         'longer_than_reference' if output_delta_s > max_duration_difference_s else None,
        'first_downbeat_s': first_downbeat_s,
        'first_downbeat_raw_lag_s': first_lag,
        'entry_flag': ('late_possible' if first_lag > tolerance_s else
                       'early_possible' if first_lag < -tolerance_s else None) if first_lag is not None else None,
        'signal': signal,
        'detections': detections,
        'raw': raw,
        'global_shift': shift,
        'corrected': corrected,
        'section_boundaries': section_boundaries,
        'reference_uncertain_intervals': uncertainty,
        'low_confidence_downbeats': [event for event in eligible if event['low_confidence']],
        'section_transitions': [
            {**bar, 'timing_status': 'paired'} for bar in summary_view['matched'] if bar['bar'] in section_changes
        ] + [
            {**bar, 'timing_status': 'unmatched'} for bar in summary_view['unmatched_reference'] if bar['bar'] in section_changes
        ],
        'analysis_seconds': perf_counter() - started,
    }


def public_detector(provenance):
    return {key: provenance[key] for key in (
        'backbone', 'postprocessor', 'activation_hz', 'torch', 'torchaudio', 'device',
        'confidence', 'low_confidence_below',
    )}


def public_report(report):
    """Copy numeric evidence and display metadata without source paths or lyrics."""
    def public_rows(rows):
        fields = {'bar', 'time_s', 'section', 'detected_s', 'raw_lag_s', 'residual_s',
                  'confidence', 'low_confidence', 'reference_uncertainty', 'timing_flag',
                  'timing_status', 'reason'}
        return [{key: value for key, value in row.items() if key in fields} for row in rows]

    public = {key: value for key, value in report.items() if key not in {
        'detector', 'raw', 'corrected', 'section_transitions', 'reference_basis',
    }}
    public['detector'] = public_detector(report['detector'])
    public['reference_uncertain_intervals'] = [
        {key: span[key] for key in ('start_s', 'end_s', 'reason')}
        for span in report['reference_uncertain_intervals']
    ]
    for key in ('raw', 'corrected'):
        source = report[key]
        public[key] = {**source, 'matched': public_rows(source['matched']),
                       'unmatched_reference': public_rows(source['unmatched_reference'])} if source is not None else None
    public['section_transitions'] = public_rows(report['section_transitions'])
    return public


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True, help='CASM MIREX 2026 bundle directory')
    parser.add_argument('--checkpoint', type=Path, help='override the bundle checkpoint')
    parser.add_argument('--config', type=Path, help='override the bundle CASM config')
    parser.add_argument('--source-commit', help='checkout revision for the internal receipt')
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--kind', choices=['backing', 'mix', 'backing_proxy', 'control'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--public-output', type=Path, help='add a sanitized timing report')
    parser.add_argument('--tolerance-s', type=float, default=0.15)
    parser.add_argument('--maximum-shift-s', type=float, default=0.75)
    args = parser.parse_args()
    if not 0 < args.tolerance_s < args.maximum_shift_s < math.inf:
        parser.error('require 0 < tolerance < maximum shift')
    bundle = args.bundle.resolve()
    detector = Detector(
        bundle,
        args.checkpoint or bundle / 'weights/beatthis_mirex.ckpt',
        args.config or bundle / 'config/casm-no-smc.json',
        args.source_commit,
    )
    report = analyze(detector, args.audio, args.reference, args.kind, args.tolerance_s, args.maximum_shift_s)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    if args.public_output:
        args.public_output.parent.mkdir(parents=True, exist_ok=True)
        args.public_output.write_text(json.dumps(public_report(report), ensure_ascii=False, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
