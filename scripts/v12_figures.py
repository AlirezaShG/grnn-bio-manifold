"""All v12 figures from results/<root>/analysis and the raw evaluation files.

python scripts/v12_figures.py --root results/v12/conf --out paper_v12/figures
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts' / 'figures'))
from style import apply_style, panel_label, TYPE_COLORS  # noqa: E402
from grnn_v1 import compare  # noqa: E402
from grnn_v1.learn import invivo_targets  # noqa: E402

RULES = {'untrained': ('untrained', '#9e9e9e'), 'readout': ('readout only', '#bdbdbd'),
         'local_nobeta': (r'local, no $\beta$', '#e07b39'), 'local': (r'local ($\beta$)', '#1f77b4'),
         'local_sym': ('local, sym. fb.', '#2ca02c'), 'bptt': ('BPTT', '#222222')}
OBJ = {'data': 'data matching', 'change': 'change detection'}


def dots(ax, df, metric, objective, rules, ref=None, ylabel=None):
    d = df[df.objective == objective]
    xs = []
    for k, r in enumerate(rules):
        v = d[d.rule == r].sort_values('seed')[metric].to_numpy(float)
        if not len(v):
            continue
        ax.scatter(np.full(len(v), k) + np.random.default_rng(k).uniform(-.12, .12, len(v)), v, s=9,
                   color=RULES[r][1], lw=0, zorder=3)
        ax.plot([k - .3, k + .3], [np.nanmean(v)] * 2, color='k', lw=1.3, zorder=4)
        xs.append(k)
    if ref is not None:
        ax.axhline(ref, color='#c0392b', lw=.8, ls='--')
    short = {'untrained': 'untr.', 'readout': 'readout', 'local_nobeta': 'no beta', 'local': 'local',
             'local_sym': 'sym.', 'bptt': 'BPTT'}
    ax.set_xticks(range(len(rules)), [short[r] for r in rules], rotation=45, ha='right')
    if ylabel:
        ax.set_ylabel(ylabel)


def fig_geometry(df, ref, out):
    metrics = [('rsa_dis_all', 'RSA with mouse\n(disattenuated)', None),
               ('cka_all', 'linear CKA with mouse', None),
               ('pr_all', 'dimensionality (PR, 200 units)', ref['pr_mouse_all']),
               ('decode_all', 'image decoding (200 units)', ref['decode_mouse_all']),
               ('sparse_sim_all', 'lifetime sparseness\nsimilarity (1-KS)', None),
               ('sim_tuning_mean', 'tuning-distribution\nsimilarity (1-KS)', None)]
    fig, axes = plt.subplots(2, len(metrics), figsize=(7.4, 4.6))
    for r, obj in enumerate(['data', 'change']):
        rules = [x for x in ['untrained', 'readout', 'local_nobeta', 'local', 'local_sym', 'bptt']
                 if len(df[(df.objective == obj) & (df.rule == x)])]
        for c, (m, lab, rf) in enumerate(metrics):
            ax = axes[r, c]
            dots(ax, df, m, obj, rules, rf, lab if r == 0 else lab)
            ax.set_title(OBJ[obj] if c == 0 else '', fontsize=7, loc='left')
            panel_label(ax, 'abcdefghijkl'[r * len(metrics) + c])
    fig.tight_layout()
    fig.savefig(out / 'fig3_geometry.pdf', bbox_inches='tight'); fig.savefig(out / 'fig3_geometry.png', dpi=200, bbox_inches='tight')


def fig_trajectory(traj, out):
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.3))
    for c, (obj, m) in enumerate([('data', 'rsa_all'), ('data', 'cka_all'), ('change', 'rsa_all'), ('change', 'cka_all')]):
        ax = axes[c]
        d = traj[traj.objective == obj]
        for r in ['local_nobeta', 'local', 'local_sym', 'bptt', 'readout']:
            s = d[d.rule == r]
            if not len(s):
                continue
            g = s.groupby('step')[['val', m]].mean()
            ax.plot(g['val'], g[m], '-o', ms=2.5, lw=1, color=RULES[r][1], label=RULES[r][0])
        ax.set_xlabel('validation loss' if obj == 'data' else 'validation accuracy')
        if obj == 'data':
            ax.invert_xaxis()
        ax.set_ylabel(m.replace('_all', '').upper() + ' with mouse')
        ax.set_title(OBJ[obj], fontsize=7)
        panel_label(ax, 'abcd'[c])
    axes[0].legend(frameon=False, fontsize=5)
    axes[2].legend(frameon=False, fontsize=5)
    fig.tight_layout()
    fig.savefig(out / 'fig4_trajectory.pdf', bbox_inches='tight'); fig.savefig(out / 'fig4_trajectory.png', dpi=200, bbox_inches='tight')


def fig_tuning(root, out, seed):
    tg = invivo_targets()
    fig, axes = plt.subplots(2, 4, figsize=(7.4, 3.6))
    for c, t in enumerate(['Pyr', 'PV', 'SST', 'VIP']):
        for r, (stat, lab) in enumerate([('pref', 'pref. rate (Hz)'), ('osi', 'OSI')]):
            ax = axes[r, c]
            data = tg[t]['pref' if stat == 'pref' else 'osi']
            ax.hist(np.log10(data + .1) if stat == 'pref' else data, bins=20, density=True, histtype='stepfilled',
                    color='#f2c4c4', label=f'mouse (n={len(data)})')
            for rule in ['untrained', 'local', 'bptt']:
                obj = 'data'
                p = root / obj / rule / str(seed) / 'eval_best.npz'
                if not p.exists():
                    continue
                ev = np.load(p, allow_pickle=True)
                i = ev['types'] == t
                v = ev['gr_pref'][i] if stat == 'pref' else ev['gr_osi'][i][ev['gr_pref'][i] >= .5]
                ax.hist(np.log10(v + .1) if stat == 'pref' else v, bins=20, density=True, histtype='step', lw=1.1,
                        color=RULES[rule][1], label=RULES[rule][0])
            ax.set_xlabel('log10(' + lab + ' + 0.1)' if stat == 'pref' else lab)
            if r == 0:
                ax.set_title(t, color=TYPE_COLORS[t])
    axes[0, 0].legend(frameon=False, fontsize=5)
    fig.tight_layout()
    fig.savefig(out / 'fig2_tuning.pdf', bbox_inches='tight'); fig.savefig(out / 'fig2_tuning.png', dpi=200, bbox_inches='tight')


def fig_rdms(root, out, seed):
    mouse = compare.load_mouse()
    R = compare.rdm(mouse['mean'])
    order = np.argsort(np.argsort(R.mean(0)))
    import scipy.cluster.hierarchy as h
    order = h.leaves_list(h.linkage(compare.zscore_units(mouse['mean']), 'average', metric='correlation'))
    panels = [('mouse V1', R)]
    for obj, rule in [('data', 'untrained'), ('data', 'local'), ('data', 'bptt'), ('change', 'local'), ('change', 'bptt')]:
        p = root / obj / rule / str(seed) / 'eval_best.npz'
        if p.exists():
            ns = np.load(p, allow_pickle=True)['ns'].astype(np.float32).mean(0)
            ns = ns[:, ns.mean(0) >= .1]
            rho = stats.spearmanr(compare.upper(compare.rdm(ns)), compare.upper(R))[0]
            panels.append((f'{OBJ[obj] if rule != "untrained" else ""} {RULES[rule][0]}\n(rho={rho:.2f})', compare.rdm(ns)))
    fig, axes = plt.subplots(1, len(panels), figsize=(7.4, 1.9))
    for ax, (t, M) in zip(axes, panels):
        ax.imshow(M[np.ix_(order, order)], cmap='viridis', vmin=np.percentile(M, 2), vmax=np.percentile(M, 98))
        ax.set_title(t, fontsize=6); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out / 'fig5_rdms.pdf', bbox_inches='tight'); fig.savefig(out / 'fig5_rdms.png', dpi=200, bbox_inches='tight')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=1001)
    a = ap.parse_args()
    apply_style()
    a.out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.root / 'analysis' / 'network_metrics.csv')
    ref = json.loads((a.root / 'analysis' / 'mouse_reference.json').read_text())
    fig_geometry(df, ref, a.out)
    tp = a.root / 'analysis' / 'trajectory_metrics.csv'
    if tp.exists() and tp.stat().st_size > 10:
        fig_trajectory(pd.read_csv(tp), a.out)
    fig_tuning(a.root, a.out, a.seed)
    fig_rdms(a.root, a.out, a.seed)
    print('figures written to', a.out)


if __name__ == '__main__':
    main()
