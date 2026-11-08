"""Figure 2: learning rules on three tasks (Family A), primary endpoint.

Usage: python scripts/figures/fig2_learning_rules.py --root results/v11/confirmation --prefix c
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


def stars(p):
    if p is None or not np.isfinite(p):
        return 'n.e.'
    return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else 'n.s.'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--prefix', required=True)
    ap.add_argument('--out', type=Path, default=ROOT / 'paper' / 'figures')
    ap.add_argument('--split', default='test_delay')
    ap.add_argument('--metric', default='accuracy_incongruent')
    ap.add_argument('--tag', default='fig2_learning_rules')
    a = ap.parse_args()
    apply_style()
    tests = json.loads((a.root / 'analysis' / 'confirmatory_tests.json').read_text())
    fam = pd.DataFrame(tests['family_A']) if (a.split, a.metric) == ('test_delay', 'accuracy_incongruent') else \
        pd.DataFrame(tests['secondary'][f'{a.split}/{a.metric}']['A'])
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), sharey=True)
    for r, opt in enumerate(['adam', 'rms']):
        for c, task in enumerate(['early', 'late', 'dms']):
            ax = axes[r, c]
            m = pd.read_csv(a.root / f'{a.prefix}_{task}_{opt}' / 'metrics.csv')
            m = m[m.split == a.split]
            table = m.pivot(index='seed', columns='condition', values=a.metric) * 100
            cols = ['intact_' + l for l in LEARNERS]
            for _, row in table[cols].iterrows():
                ax.plot(range(4), row.values, color='#bbbbbb', lw=.4, zorder=1)
            for j, l in enumerate(LEARNERS):
                v = table['intact_' + l].values
                ax.scatter(np.full(len(v), j), v, s=7, color=LEARNER_COLORS[l], zorder=2, lw=0)
                ax.plot([j - .28, j + .28], [v.mean()] * 2, color='k', lw=1.4, zorder=3)
            ax.axhline(50, color='#999999', lw=.6, ls='--')
            ax.set_xticks(range(4), [LEARNER_LABELS[l] for l in LEARNERS], rotation=30, ha='right')
            ax.set_title(f'{TASK_LABELS[task]} | {"Adam" if opt == "adam" else "RMS (local)"}')
            if c == 0:
                ax.set_ylabel('unseen-delay conflict accuracy (%)' if a.metric == 'accuracy_incongruent'
                              else f'{a.split} {a.metric} (%)')
            f = fam[(fam.task == task) & (fam.optimizer == opt)]
            y0 = 101
            for label, (x0, x1) in [('A3', (0, 1)), ('A1', (1, 2)), ('A2', (1, 3))]:
                rr = f[f.contrast.str.startswith(label)]
                if len(rr):
                    p = rr.iloc[0].get('p_holm', rr.iloc[0].get('p_value'))
                    ax.plot([x0, x1], [y0, y0], color='k', lw=.6)
                    ax.text((x0 + x1) / 2, y0 + .5, stars(p), ha='center', va='bottom', fontsize=6)
                    y0 += 5
            ax.set_ylim(35, 118)
            ax.set_yticks([40, 50, 60, 70, 80, 90, 100])
            panel_label(ax, 'abcdef'[r * 3 + c])
    fig.tight_layout()
    a.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out / f'{a.tag}.pdf', bbox_inches='tight')
    fig.savefig(a.out / f'{a.tag}.png', dpi=200, bbox_inches='tight')
    print('saved', a.out / f'{a.tag}.pdf')


if __name__ == '__main__':
    main()
