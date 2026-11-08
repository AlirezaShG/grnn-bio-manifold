"""Figure 6: operating point (Family D). Accuracy and rate vs the calibrated baseline fraction."""
import argparse
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
LEVELS = [(0.02, 'op02'), (0.05, 'intact'), (0.10, 'op10'), (0.20, 'op20')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--prefix', required=True)
    ap.add_argument('--out', type=Path, default=ROOT / 'paper' / 'figures')
    a = ap.parse_args()
    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.0), sharex=True)
    for c, task in enumerate(['early', 'late', 'dms']):
        m = pd.concat([pd.read_csv(a.root / f'{a.prefix}_{task}_adam' / 'metrics.csv'),
                       pd.read_csv(a.root / f'{a.prefix}D_{task}_adam' / 'metrics.csv')])
        m = m[m.split == 'test_delay']
        for learner in ['bptt', 'local']:
            xs, acc, rate = [], [], []
            for f, p in LEVELS:
                name = f'intact_{learner}' if p == 'intact' else f'{p}_{learner}'
                s = m[m.condition == name].sort_values('seed')
                xs.append(100 * f)
                acc.append(100 * s.accuracy_incongruent.to_numpy())
                rate.append(s.mean_rate_hz.to_numpy())
            for ax, vals in [(axes[0, c], acc), (axes[1, c], rate)]:
                mu = [v.mean() for v in vals]
                se = [v.std(ddof=1) / np.sqrt(len(v)) for v in vals]
                ax.errorbar(xs, mu, yerr=se, color=LEARNER_COLORS[learner], marker='o', ms=3, lw=1,
                            label=LEARNER_LABELS[learner])
        axes[0, c].set_title(TASK_LABELS[task])
        axes[0, c].axhline(50, color='#999999', lw=.6, ls='--')
        axes[1, c].set_xscale('log')
        axes[1, c].set_xticks([2, 5, 10, 20], ['2', '5', '10', '20'])
        axes[1, c].set_xlabel('baseline (% of fitted ceiling)')
        panel_label(axes[0, c], 'abc'[c])
        panel_label(axes[1, c], 'def'[c])
    axes[0, 0].set_ylabel('unseen-delay conflict acc. (%)')
    axes[1, 0].set_ylabel('mean rate (Hz)')
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    a.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out / 'fig6_operating_point.pdf', bbox_inches='tight')
    fig.savefig(a.out / 'fig6_operating_point.png', dpi=200, bbox_inches='tight')
    print('saved')


if __name__ == '__main__':
    main()
