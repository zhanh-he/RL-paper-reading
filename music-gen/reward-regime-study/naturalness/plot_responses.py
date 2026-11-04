"""Plot measured paired scorer responses, never assumed human degradation labels.

Reproduce with Python 3.11 and matplotlib 3.11.2:
    python plot_responses.py

Defaults read the published naturalness audit and regenerate its PNG/SVG figures.
Each dot is one known operation on one fixed 10-second source, without uncertainty
bars: two source examples cannot estimate population-level confidence intervals.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OPERATIONS = [
    ('gain_minus20db', 'Gain -20 dB\nvs original'),
    ('lowpass_3khz', 'Low-pass 3 kHz\nvs original'),
    ('compressed', 'Compressor\nvs original'),
    ('distorted', 'Hard clipping\nvs original'),
    ('metallic_mod40hz', '40 Hz AM\nvs RMS-matched gain'),
    ('mod40hz_gain_control', 'AM matched-gain control\nvs original'),
]
SOURCES = [('muse_baseline', 'Muse baseline', '#216d8f', 'o'),
           ('ref_vibeace', 'VibeAce recording', '#ab542c', 's')]
METRICS = [('audiobox', 'PQ', 'AudioBox PQ change', 'Delta PQ (score units)'),
           ('pam', 'score', 'Official PAM change', 'Delta PAM (score units)')]


def responses(audit: dict) -> dict:
    samples = {row['id']: row for row in audit['samples']}
    if len(samples) != len(audit['samples']):
        raise ValueError('Duplicate sample IDs')
    recorded = {row['id']: row for row in audit.get('comparisons', []) if row.get('id')}
    values = {}
    for source, _, _, _ in SOURCES:
        values[source] = []
        for operation, _ in OPERATIONS:
            item = samples[f'{source}__{operation}']
            comparator_id = (f'{source}__mod40hz_gain_control'
                             if operation == 'metallic_mod40hz' else source)
            if item.get('matched_comparator_id') != comparator_id:
                raise ValueError(f'Unexpected comparator for {item["id"]}')
            comparator = samples[comparator_id]
            delta = []
            for family, head, _, _ in METRICS:
                score = float(item[family][head])
                reference = float(comparator[family][head])
                if not math.isfinite(score) or not math.isfinite(reference):
                    raise ValueError('A missing/nonfinite score must not become zero')
                delta.append(score-reference)
            receipt = recorded.get(item['id'])
            if receipt:
                for key, value in zip(['pq_delta', 'pam_delta'], delta):
                    if not math.isclose(float(receipt[key]), value, abs_tol=1e-8):
                        raise ValueError(f'Published comparison mismatch: {item["id"]}/{key}')
            values[source].append(delta)
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path,
                        default=ROOT/'platform/site/demos/data/naturalness-audit.json')
    parser.add_argument('--output-stem', type=Path,
                        default=ROOT/'platform/site/demos/visuals/naturalness-response')
    args = parser.parse_args()
    raw = args.audit.read_bytes()
    audit = json.loads(raw)
    values = responses(audit)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.spines.left': False, 'svg.hashsalt': 'naturalness-response-v1'})
    fig, axes = plt.subplots(1, 2, figsize=(13.7, 5.5), sharey=True,
                             gridspec_kw={'wspace': .22})
    positions = np.arange(len(OPERATIONS))
    for metric_index, (axis, (_, _, title, xlabel)) in enumerate(zip(axes, METRICS)):
        flat = [value[metric_index] for source in values.values() for value in source]
        left, right = min(flat+[0]), max(flat+[0])
        span = max(right-left, .1)
        axis.set_xlim(left-.18*span, right+.22*span)
        axis.axvline(0, color='#444b50', linewidth=1.0, zorder=1)
        axis.grid(axis='x', color='#d8dfe2', linewidth=.65, alpha=.7)
        axis.set_axisbelow(True)
        for source_index, (source, label, color, marker) in enumerate(SOURCES):
            y = positions+(-.145 if source_index == 0 else .145)
            x = np.array(values[source])[:, metric_index]
            axis.hlines(y, 0, x, color=color, alpha=.38, linewidth=1.6, zorder=2)
            axis.scatter(x, y, s=44, color=color, marker=marker, label=label, zorder=3)
            for xx, yy in zip(x, y):
                axis.annotate(f'{xx:+.3f}', (xx, yy), xytext=(7 if xx >= 0 else -7, 0),
                              textcoords='offset points', va='center',
                              ha='left' if xx >= 0 else 'right', fontsize=8.6, color=color)
        axis.set_title(f'({"ab"[metric_index]}) {title}', loc='left', fontsize=12, fontweight='bold', pad=16)
        axis.set_xlabel(xlabel, labelpad=10)
        axis.tick_params(axis='y', length=0, pad=12)
        axis.spines['bottom'].set_color('#bbc4c9')
        axis.set_yticks(positions, [title for _, title in OPERATIONS])
    axes[0].invert_yaxis()
    axes[0].set_ylim(len(OPERATIONS)-.55, -.55)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.61, .925),
               ncol=2, frameon=False, handletextpad=.55, columnspacing=2.4)
    fig.suptitle('Measured responses to controlled audio operations', x=.12, ha='left',
                 y=.995, fontsize=15, fontweight='bold')
    fig.text(.12, .946, 'Two fixed 10 s sources | independent horizontal scales | zero = no change',
             fontsize=10, color='#555e63')
    fig.subplots_adjust(left=.205, right=.985, top=.79, bottom=.18)
    fig.text(.12, .065, 'Known operations are not human defect labels. AM uses an equal-RMS gain control; other rows use the original.',
             fontsize=9.1, color='#555e63')
    fig.text(.12, .026, f'Source: naturalness-audit.json | {audit["metadata"]["version"]} | human ratings: {audit["metadata"]["human_labels_count"]}',
             fontsize=8.5, color='#687278')
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    description = ('Paired scorer responses; not human naturalness judgments. '
                   f'Audit SHA256 {hashlib.sha256(raw).hexdigest()}')
    fig.savefig(args.output_stem.with_suffix('.png'), dpi=180,
                metadata={'Software': 'plot_responses.py; matplotlib', 'Description': description})
    fig.savefig(args.output_stem.with_suffix('.svg'),
                metadata={'Creator': 'plot_responses.py; matplotlib', 'Date': None,
                          'Description': description})
    svg = args.output_stem.with_suffix('.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)
    print(json.dumps({'audit_sha256': hashlib.sha256(raw).hexdigest(),
                      'source_count': len(SOURCES), 'operation_count': len(OPERATIONS),
                      'outputs': [str(args.output_stem.with_suffix(ext)) for ext in ['.png', '.svg']]}))


if __name__ == '__main__':
    main()
