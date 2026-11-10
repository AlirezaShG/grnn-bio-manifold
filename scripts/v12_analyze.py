"""Compare every trained network with mouse V1 and test rule differences.

python scripts/v12_analyze.py --root results/v12/conf --out results/v12/conf/analysis
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grnn_v1 import compare  # noqa: E402
from grnn_v1.learn import invivo_targets  # noqa: E402
from grnn_v1.stats import paired_comparison, holm_adjust  # noqa: E402

MOUSE = None


def _init():
    global MOUSE, TARGETS
    MOUSE = compare.load_mouse()
    TARGETS = invivo_targets()


def one(path: str):
    p = Path(path)
    ev = dict(np.load(p / 'eval_best.npz', allow_pickle=True))
    seed = int(p.name)
    row = {'objective': p.parent.parent.name, 'rule': p.parent.name, 'seed': seed,
           **compare.compare_network(ev, MOUSE, TARGETS, seed=seed)}
    done = json.loads((p / 'done.json').read_text())
    row.update({'best_val': done.get('best_val'), 'best_step': done.get('best_step')})
    traj = []
    if (p / 'eval_traj.npz').exists():
        t = np.load(p / 'eval_traj.npz', allow_pickle=True)
        types = t['types']
        for k, v in zip(t['steps'], t['vals']):
            ns = t[str(k)].astype(np.float32)
            act = ns.mean((0, 1)) >= 0.1
            mm = ns[:, :, act].mean(0)
            rho = stats.spearmanr(compare.upper(compare.rdm(mm)), compare.upper(compare.rdm(MOUSE['mean'])))[0]
            traj.append({'objective': row['objective'], 'rule': row['rule'], 'seed': seed,
                         'step': int(str(k).split('_')[1]), 'val': float(v), 'rsa_all': float(rho),
                         'cka_all': compare.linear_cka(compare.zscore_units(mm), compare.zscore_units(MOUSE['mean']))})
    return row, traj


def paired(df, metric, a, b, objective):
    x = df[(df.objective == objective) & (df.rule == a)].set_index('seed')[metric]
    y = df[(df.objective == objective) & (df.rule == b)].set_index('seed')[metric]
    s = sorted(set(x.index) & set(y.index))
    if len(s) < 2:
        return None
    r = paired_comparison(x.loc[s].to_numpy(float), y.loc[s].to_numpy(float), seed=7)
    return {'objective': objective, 'metric': metric, 'a': a, 'b': b, 'n': len(s),
            'mean_a': r['mean_a'], 'mean_b': r['mean_b'], 'diff': r['mean_difference_a_minus_b'],
            'ci_low': r['ci_low'], 'ci_high': r['ci_high'], 'dz': r['effect_dz'], 'p': r['p_value'],
            'wins_a': int(np.sum(x.loc[s].to_numpy() > y.loc[s].to_numpy()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--untrained_root', type=Path, default=None)
    ap.add_argument('--jobs', type=int, default=12)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    runs = sorted(str(p.parent) for p in a.root.glob('*/*/*/eval_best.npz'))
    with ProcessPoolExecutor(a.jobs, initializer=_init) as pool:
        res = list(pool.map(one, runs))
    df = pd.DataFrame([r for r, _ in res])
    traj = pd.DataFrame([t for _, ts in res for t in ts])
    # the untrained network is the same circuit for both objectives
    un = df[df.rule == 'untrained']
    for obj in ('data', 'change'):
        if not len(df[(df.objective == obj) & (df.rule == 'untrained')]) and len(un):
            df = pd.concat([df, un.assign(objective=obj)], ignore_index=True)
    df.to_csv(a.out / 'network_metrics.csv', index=False)
    traj.to_csv(a.out / 'trajectory_metrics.csv', index=False)
    _init()
    ceil = compare.mouse_ceilings(MOUSE)
    json.dump(ceil, open(a.out / 'mouse_reference.json', 'w'), indent=2)
    primary = ['rsa_dis_all']
    secondary = ['rsa_all', 'cka_all', 'pr_all', 'sparse_sim_all', 'decode_all', 'sim_tuning_mean', 'rsa_dis_exc', 'cka_exc']
    tests = []
    for obj in ('data', 'change'):
        fam = []
        for m in primary + secondary:
            for x, y in [('local', 'bptt'), ('local', 'local_nobeta'), ('bptt', 'untrained'), ('local', 'untrained'),
                         ('local_nobeta', 'untrained'), ('local', 'readout'), ('bptt', 'readout'), ('local_sym', 'local')]:
                r = paired(df, m, x, y, obj)
                if r:
                    r['family'] = 'primary' if m in primary else 'secondary'
                    fam.append(r)
        for fam_name in ('primary', 'secondary'):
            sub = [r for r in fam if r['family'] == fam_name]
            for r, ph in zip(sub, holm_adjust([r['p'] for r in sub])):
                r['p_holm'] = ph
        tests.extend(fam)
    pd.DataFrame(tests).to_csv(a.out / 'paired_tests.csv', index=False)
    summary = df.groupby(['objective', 'rule'])[[c for c in df.columns if c not in ('objective', 'rule', 'seed')]].mean()
    summary.to_csv(a.out / 'condition_means.csv')
    pd.set_option('display.width', 220)
    show = ['rsa_all', 'rsa_dis_all', 'cka_all', 'pr_all', 'pr_mouse_all', 'sparse_sim_all', 'decode_all', 'sim_tuning_mean', 'best_val']
    print(summary[[c for c in show if c in summary]].round(3).to_string())
    print('mouse reference', {k: round(v, 3) if isinstance(v, float) else v for k, v in ceil.items()})
    t = pd.DataFrame(tests)
    if len(t):
        print(t[t.family == 'primary'][['objective', 'metric', 'a', 'b', 'mean_a', 'mean_b', 'diff', 'ci_low', 'ci_high', 'p', 'p_holm', 'wins_a']].round(4).to_string(index=False))


if __name__ == '__main__':
    main()
