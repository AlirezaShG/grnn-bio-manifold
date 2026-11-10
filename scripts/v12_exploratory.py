"""Exploratory (not in the locked protocol): how far did learning move the circuit?

Per network: relative recurrent weight change ||W_best - W_0|| / ||W_0||, and the
Spearman correlation between the network's own trained and untrained natural-scene
RDMs (1 = geometry unchanged). Also the per-type change in mean natural-scene rate.
python scripts/v12_exploratory.py --root results/v12/conf
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grnn_v1 import compare  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    a = ap.parse_args()
    rows = []
    for p in sorted(a.root.glob('*/*/*/eval_best.npz')):
        obj, rule, seed = p.parts[-4], p.parts[-3], int(p.parts[-2])
        if rule == 'untrained':
            continue
        un = a.root / 'data' / 'untrained' / str(seed) / 'eval_best.npz'
        if not un.exists():
            continue
        ck = np.load(p.parent / 'checkpoints.npz')
        w0, wb = ck['step_0__mag'], ck['best__mag']
        ev, e0 = np.load(p, allow_pickle=True), np.load(un, allow_pickle=True)
        m, m0 = ev['ns'].astype(np.float32).mean(0), e0['ns'].astype(np.float32).mean(0)
        act = (m.mean(0) >= .1) & (m0.mean(0) >= .1)
        r = stats.spearmanr(compare.upper(compare.rdm(m[:, act])), compare.upper(compare.rdm(m0[:, act])))[0]
        row = {'objective': obj, 'rule': rule, 'seed': seed,
               'rel_weight_change': float(np.linalg.norm(wb - w0) / np.linalg.norm(w0)),
               'rdm_vs_untrained': float(r)}
        for t in ('Pyr', 'PV', 'SST', 'VIP'):
            i = ev['types'] == t
            row[f'ns_rate_{t}'] = float(m[:, i].mean())
            row[f'ns_rate_change_{t}'] = float(m[:, i].mean() - m0[:, i].mean())
        rows.append(row)
    df = pd.DataFrame(rows)
    out = a.root / 'analysis'
    out.mkdir(exist_ok=True)
    df.to_csv(out / 'exploratory_movement.csv', index=False)
    pd.set_option('display.width', 200)
    print(df.groupby(['objective', 'rule'])[['rel_weight_change', 'rdm_vs_untrained', 'ns_rate_Pyr', 'ns_rate_PV']].agg(['mean', 'std']).round(3).to_string())


if __name__ == '__main__':
    main()
