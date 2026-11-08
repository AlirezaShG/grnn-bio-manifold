"""Figure 3: why the spike-history term matters (credit-assignment alignment).

(a) cosine(local estimate, exact gradient) over training, in networks trained with the local
    rule; with/without beta, symmetric/random feedback (Adam, each task).
(b) per-neuron alignment gain from beta (symmetric feedback, Pyr cells) vs adaptation loop gain.
(c) per-type alignment with random feedback, with vs without beta.
Inputs: suites <prefix>_<task>_adam and a neuron_alignment CSV made by
scripts/analysis/neuron_alignment.py.
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
from style import apply_style, TYPE_COLORS, TASK_LABELS, panel_label  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SERIES = [('grad_local_symmetric_cosine', r'$\beta$, symmetric', '#1f77b4', '-'),
          ('grad_no_beta_symmetric_cosine', r'no $\beta$, symmetric', '#e07b39', '-'),
          ('grad_local_random_cosine', r'$\beta$, random', '#1f77b4', '--'),
          ('grad_no_beta_random_cosine', r'no $\beta$, random', '#e07b39', '--')]


def checkpoint_table(suite, condition):
    rows = []
    for p in sorted((suite / 'runs' / condition).glob('*/checkpoint_geometry.json')):
        for r in json.loads(p.read_text()):
            rows.append({'seed': p.parent.name, 'step': r['step'], **{k: r.get(k) for k, *_ in SERIES}})
    return pd.DataFrame(rows)


def network_corr(df, x, y):
    zs = []
    for _, g in df.groupby('seed'):
        g = g[np.isfinite(g[x]) & np.isfinite(g[y])]
        if len(g) > 4 and g[x].std() > 0 and g[y].std() > 0:
            zs.append(np.arctanh(np.clip(np.corrcoef(g[x], g[y])[0, 1], -.999, .999)))
    zs = np.array(zs)
    if not len(zs):
        return np.nan, np.nan, np.nan, 0
    boot = np.random.default_rng(0).choice(zs, (5000, len(zs))).mean(1)
    return np.tanh(zs.mean()), np.tanh(np.quantile(boot, .025)), np.tanh(np.quantile(boot, .975)), len(zs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--prefix', required=True)
    ap.add_argument('--neurons', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=ROOT / 'paper' / 'figures')
    ap.add_argument('--tag', default='fig3_mechanism')
    a = ap.parse_args()
    apply_style()
    fig = plt.figure(figsize=(7.2, 4.8))
    gs = fig.add_gridspec(2, 3, hspace=.6, wspace=.42)
    stats = {}
    for c, task in enumerate(['early', 'late', 'dms']):
        ax = fig.add_subplot(gs[0, c])
        t = checkpoint_table(a.root / f'{a.prefix}_{task}_adam', 'intact_local')
        for key, label, color, ls in SERIES:
            g = t.groupby('step')[key]
            mu, se = g.mean(), g.std() / np.sqrt(g.count())
            ax.plot(mu.index, mu.values, color=color, ls=ls, lw=1.1, label=label)
            ax.fill_between(mu.index, mu - se, mu + se, color=color, alpha=.15, lw=0)
        ax.set_title(f'{TASK_LABELS[task]} (local-trained)')
        ax.set_xlabel('update'); ax.set_ylim(-.1, 1)
        if c == 0:
            ax.set_ylabel('cosine with exact gradient')
            ax.legend(frameon=False, fontsize=5.5, loc='lower right')
        panel_label(ax, 'abc'[c])
    nd = pd.read_csv(a.neurons)
    nd = nd[(nd.condition == 'intact_local')]
    nd['gain_beta'] = nd.cos_beta_symmetric - nd.cos_nobeta_symmetric
    ax = fig.add_subplot(gs[1, 0:2])
    pyr = nd[nd.cell_type == 'Pyr'].copy()
    for task, mk in zip(['early', 'late', 'dms'], ['o', 's', '^']):
        s = pyr[pyr.suite.str.endswith(f'{task}_adam')]
        ax.scatter(s.adaptation_loop_gain, s.gain_beta, s=3, alpha=.35, marker=mk, lw=0,
                   color={'early': '#1f77b4', 'late': '#2ca02c', 'dms': '#9467bd'}[task], label=TASK_LABELS[task])
        r, lo, hi, n = network_corr(s.assign(seed=s.suite + s.seed.astype(str)), 'adaptation_loop_gain', 'gain_beta')
        stats[task] = {'r': r, 'ci': [lo, hi], 'n_networks': n}
    ax.axhline(0, color='#999999', lw=.6)
    ax.set_xscale('log')
    ax.set_xlabel(r'adaptation loop gain $|\langle g^\prime\rangle\bar\beta|$ (Pyr cells)')
    ax.set_ylabel(r'alignment gain from $\beta$' + '\n(cos with $\\beta$ - cos without)')
    txt = '; '.join(f"{TASK_LABELS[k]} r={v['r']:.2f} [{v['ci'][0]:.2f},{v['ci'][1]:.2f}]" for k, v in stats.items())
    ax.set_title('per-network correlation: ' + txt, fontsize=6)
    ax.legend(frameon=False, markerscale=3)
    panel_label(ax, 'd')
    ax = fig.add_subplot(gs[1, 2])
    types = ['Pyr', 'PV', 'SST', 'VIP']
    agg = nd.groupby(['suite', 'seed', 'cell_type'])[['cos_beta_random', 'cos_nobeta_random']].mean().reset_index()
    for j, t in enumerate(types):
        s = agg[agg.cell_type == t]
        for k, (col, off, face) in enumerate([('cos_beta_random', -.17, TYPE_COLORS[t]), ('cos_nobeta_random', .17, 'white')]):
            v = s[col].values.astype(float)
            v = v[np.isfinite(v)]
            if not len(v):
                continue
            ax.errorbar(j + off, v.mean(), yerr=v.std() / np.sqrt(len(v)), fmt='o', ms=3.5, color=TYPE_COLORS[t],
                        mfc=face, lw=.8)
    ax.set_xticks(range(4), types); ax.axhline(0, color='#999999', lw=.6)
    ax.set_ylabel('row cosine, random feedback')
    ax.set_title(r'filled: with $\beta$; open: without', fontsize=6)
    panel_label(ax, 'e')
    a.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out / f'{a.tag}.pdf', bbox_inches='tight')
    fig.savefig(a.out / f'{a.tag}.png', dpi=200, bbox_inches='tight')
    (a.out / f'{a.tag}_stats.json').write_text(json.dumps(stats, indent=2, default=float))
    print('saved', a.out / f'{a.tag}.pdf', stats)


if __name__ == '__main__':
    main()
