"""Figure 1: tasks and the data-fitted cells (development cohort only).

(a-c) one example trial per task; (d) mean spike-history kernel per type;
(e) steady-state history gain per cell; (f) calibrated isolated baseline rate.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import apply_style, TYPE_COLORS, panel_label  # noqa: E402
from grnn_bio.data import load_dataset, partition_cells, parameters_from_rows  # noqa: E402
from grnn_bio.model import calibrate_current  # noqa: E402
from grnn_bio.task import TaskConfig, generate_batch  # noqa: E402

apply_style()
OUT = ROOT / 'paper' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

fig = plt.figure(figsize=(7.2, 5.4))
gs = fig.add_gridspec(2, 3, hspace=.55, wspace=.45)
labels = ['stream A', 'stream B', 'cue / probe 1', 'cue / probe 2', 'go']
colors = ['#1f6f8b', '#c06c2b', '#5b8c5a', '#8a5a9e', '#333333']
tasks = [('T1-early', TaskConfig()), ('T1-late', TaskConfig(context_timing='late')),
         ('T2-dms', TaskConfig(task_type='dms', context_timing='late'))]
for i, (name, cfg) in enumerate(tasks):
    ax = fig.add_subplot(gs[0, i])
    b = generate_batch(cfg, 8, np.random.default_rng(3), delay_bins=15)
    k = int(np.flatnonzero(~b.meta['is_congruent'])[0])
    t = np.arange(b.n_steps) * 20
    for c in range(5):
        ax.plot(t, b.X[k, :, c] + 2.2 * (4 - c), color=colors[c], lw=1)
        if i == 0:
            ax.text(-25, 2.2 * (4 - c), labels[c], ha='right', va='center', fontsize=6, color=colors[c])
    for mask, col in [(b.delay_mask[k], '#eeeeee'), (b.decision_mask[k], '#fde9c9')]:
        idx = np.flatnonzero(mask)
        ax.axvspan(idx[0] * 20, (idx[-1] + 1) * 20, color=col, zorder=0)
    ax.set_yticks([]); ax.set_xlabel('time (ms)')
    tgt = '+1' if b.y[k] > 0 else '-1'
    ax.set_title(f'{name}  (target {tgt})', fontsize=8)
    ax.spines[['left']].set_visible(False)
    panel_label(ax, 'abc'[i])

df = partition_cells(load_dataset(ROOT / 'model/gfr_dataset.json', ROOT / 'data/metadata.csv'), 'development',
                     holdout_fraction=.5, seed=1701, unit='donor')
p = parameters_from_rows(df)
types = df.cell_type.to_numpy()
lags = np.arange(1, 101)
kernel = np.stack([(p.beta * (1 - p.ds) ** (l - 1)).mean(axis=1) for l in lags], axis=1)  # pA/Hz per lag
ax = fig.add_subplot(gs[1, 0])
for t in ['Pyr', 'PV', 'SST', 'VIP']:
    m = types == t
    ax.plot(lags * 20, np.median(kernel[m], axis=0), color=TYPE_COLORS[t], label=t, lw=1.2)
ax.axhline(0, color='#999999', lw=.6)
ax.set_xscale('log'); ax.set_xlabel('lag (ms)'); ax.set_ylabel('history kernel (pA/Hz)')
ax.legend(frameon=False, fontsize=6)
panel_label(ax, 'd')

gain = (p.beta / p.ds).mean(axis=1)
ax = fig.add_subplot(gs[1, 1])
rng = np.random.default_rng(0)
for j, t in enumerate(['Pyr', 'PV', 'SST', 'VIP']):
    v = gain[types == t]
    ax.scatter(j + rng.uniform(-.18, .18, len(v)), v, s=4, color=TYPE_COLORS[t], alpha=.6, lw=0)
    ax.plot([j - .25, j + .25], [np.median(v)] * 2, color='k', lw=1.2)
ax.set_xticks(range(4), ['Pyr', 'PV', 'SST', 'VIP']); ax.axhline(0, color='#999999', lw=.6)
ax.set_ylabel(r'history gain $\bar\beta$ (pA/Hz)'); ax.set_ylim(np.quantile(gain, .01) * 1.1, 2)
panel_label(ax, 'e')

offset, scale, info = calibrate_current(p)
base = np.asarray(info['achieved_rate_hz'])
ax = fig.add_subplot(gs[1, 2])
for j, t in enumerate(['Pyr', 'PV', 'SST', 'VIP']):
    v = base[types == t]
    ax.scatter(j + rng.uniform(-.18, .18, len(v)), v, s=4, color=TYPE_COLORS[t], alpha=.6, lw=0)
    ax.plot([j - .25, j + .25], [np.median(v)] * 2, color='k', lw=1.2)
ax.set_xticks(range(4), ['Pyr', 'PV', 'SST', 'VIP'])
ax.set_ylabel('isolated baseline (Hz)')
ax.set_title('set to 5% of each fitted ceiling', fontsize=7)
panel_label(ax, 'f')

fig.savefig(OUT / 'fig1_tasks_cells.pdf', bbox_inches='tight')
fig.savefig(OUT / 'fig1_tasks_cells.png', dpi=200, bbox_inches='tight')
print('saved', OUT / 'fig1_tasks_cells.pdf')
