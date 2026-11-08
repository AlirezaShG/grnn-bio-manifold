"""Figure 5: what the delay-period population holds (held-out decoders, Adam).

For each task and learner: decoding accuracy on analysis-evaluation trials of
stream A sign, stream B sign and choice in the delay phase, and of choice in the
response phase. Decoders are fit on independent analysis-fit trials.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import apply_style, LEARNER_COLORS, LEARNER_LABELS, TASK_LABELS, panel_label  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
LEARNERS = ['bptt', 'local', 'local_no_beta', 'readout_only']


def load(suite, condition):
    rows = []
    for p in sorted((suite / 'runs' / condition).glob('*/population_analysis.json')):
        for r in json.loads(p.read_text()):
            rows.append({'seed': p.parent.name, **r})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--prefix', required=True)
    ap.add_argument('--out', type=Path, default=ROOT / 'paper' / 'figures')
    ap.add_argument('--tag', default='fig5_representation')
    a = ap.parse_args()
    apply_style()
    names = {'early': ('relevant/irrelevant', 'A', 'B'), 'late': ('', 'A', 'B'), 'dms': ('', 'sample', 'distractor')}
    measures = [('delay', 'motion_dir_decoder_accuracy', 'delay: {A}'), ('delay', 'color_dir_decoder_accuracy', 'delay: {B}'),
                ('delay', 'choice_decoder_accuracy', 'delay: choice'), ('decision', 'choice_decoder_accuracy', 'response: choice')]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    summary = {}
    for c, task in enumerate(['early', 'late', 'dms']):
        ax = axes[c]
        suite = a.root / f'{a.prefix}_{task}_adam'
        width = .2
        for j, l in enumerate(LEARNERS):
            d = load(suite, 'intact_' + l)
            for k, (phase, col, lab) in enumerate(measures):
                v = 100 * d[d.phase == phase][col].to_numpy(dtype=float)
                x = k + (j - 1.5) * width
                ax.bar(x, np.nanmean(v), width * .9, color=LEARNER_COLORS[l], alpha=.85,
                       label=LEARNER_LABELS[l] if (c == 0 and k == 0) else None)
                ax.scatter(np.full(len(v), x), v, s=2, color='k', lw=0, zorder=3)
                summary[f'{task}/{l}/{phase}/{col}'] = float(np.nanmean(v))
        _, A, B = names[task]
        ax.set_xticks(range(4), [m[2].format(A=A, B=B) for m in measures], rotation=30, ha='right')
        ax.axhline(50, color='#999999', lw=.6, ls='--')
        ax.set_ylim(40, 101)
        ax.set_title(TASK_LABELS[task])
        if c == 0:
            ax.set_ylabel('held-out decoding accuracy (%)')
        panel_label(ax, 'abc'[c])
    axes[0].legend(frameon=False, fontsize=5.5, loc='lower left')
    fig.tight_layout()
    a.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out / f'{a.tag}.pdf', bbox_inches='tight')
    fig.savefig(a.out / f'{a.tag}.png', dpi=200, bbox_inches='tight')
    (a.out / f'{a.tag}_values.json').write_text(json.dumps(summary, indent=1))
    print('saved', a.out / f'{a.tag}.pdf')


if __name__ == '__main__':
    main()
