"""Print compact per-condition summaries for one or more finished suites.

Usage: python scripts/summarize_suites.py results/v11/dev/d0_budget_*  [--curves]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def summarize(suite: Path, curves: bool):
    m = pd.read_csv(suite / 'metrics.csv')
    t = pd.read_csv(suite / 'training_summary.csv')
    rows = []
    for c, g in m.groupby('condition', sort=False):
        r = {'condition': c, 'n': g.seed.nunique()}
        for split, label in [('test_id', 'ID'), ('test_delay', 'OOD'), ('test_combined', 'COMB')]:
            s = g[g.split == split]
            r[label] = f"{100*s.accuracy.mean():5.1f}±{100*s.accuracy.std():4.1f}"
            r[label + '_conf'] = f"{100*s.accuracy_incongruent.mean():5.1f}±{100*s.accuracy_incongruent.std():4.1f}"
        d = g[g.split == 'test_delay']
        r['rateHz'] = round(d.mean_rate_hz.mean(), 1)
        r['ceil%'] = round(100 * d.ceiling_rate_fraction.mean(), 2)
        tt = t[t.condition == c]
        r['best_step'] = '/'.join(str(int(x)) for x in tt.best_step)
        r['train_s'] = round(tt.training_seconds.mean())
        rows.append(r)
    print(f'\n== {suite.name}')
    print(pd.DataFrame(rows).to_string(index=False))
    if curves:
        for c in m.condition.unique():
            tabs = []
            for p in sorted((suite / 'runs' / c).glob('*/learning_curve.csv')):
                df = pd.read_csv(p).dropna(subset=['validation_bce']).set_index('step')
                tabs.append(df['validation_accuracy'])
            tab = pd.concat(tabs, axis=1).mean(axis=1)
            pick = [s for s in [50, 250, 500, 1000, 1500, 2000, 2500, 3000, 4000] if s in tab.index]
            print(f'  {c:28s} val acc: ' + ' '.join(f'{s}:{100*tab[s]:.0f}' for s in pick))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('suites', nargs='+', type=Path)
    ap.add_argument('--curves', action='store_true')
    a = ap.parse_args()
    for s in a.suites:
        if (s / 'metrics.csv').exists():
            summarize(s, a.curves)
