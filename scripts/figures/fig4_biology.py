"""Figure 4: biological identity, heterogeneity and wiring (Family B), plus controls (Family C).

(a) forest plot of Family B effects (pp, 95% CI) per task, Adam and RMS, with +/-5 pp band;
(b) operating point: delay-phase mean rate vs accuracy for intact / shuffle(reference) /
    shuffle(matched) / homogeneous / cross-type / fast decay (local learner, Adam).
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
from style import apply_style, TASK_LABELS, panel_label  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TERMS = [('B factorial: identity', 'identity (intact - shuffle)'),
         ('B factorial: topology', 'wiring (type prior - rewired)'),
         ('B factorial: identity x learner', 'identity x learner'),
         ('B factorial: topology x learner', 'wiring x learner'),
         ('B homogeneity (local): intact - homog', 'heterogeneity, local'),
         ('B homogeneity (bptt): intact - homog', 'heterogeneity, BPTT')]
CONTROLS = [('intact_local', 'intact', '#1f77b4'), ('shuffle_local', 'shuffle (matched)', '#2ca02c'),
            ('shuffle_ref_local', 'shuffle (reference)', '#d62728'), ('homog_local', 'type-mean (matched)', '#9467bd'),
            ('crosstype_local', 'cross-type (matched)', '#8c564b'), ('fastdecay_local', r'fast decay ($\tau\leq$20 ms)', '#7f7f7f')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--prefix', required=True)
    ap.add_argument('--out', type=Path, default=ROOT / 'paper' / 'figures')
    ap.add_argument('--tag', default='fig4_biology')
    a = ap.parse_args()
    apply_style()
    tests = json.loads((a.root / 'analysis' / 'confirmatory_tests.json').read_text())
    fb = pd.DataFrame(tests['family_B'])
    fig = plt.figure(figsize=(7.2, 5.2))
    gs = fig.add_gridspec(2, 3, hspace=.55, wspace=.35, height_ratios=[1.25, 1])
    for c, task in enumerate(['early', 'late', 'dms']):
        ax = fig.add_subplot(gs[0, c])
        ax.axvspan(-5, 5, color='#eeeeee', zorder=0)
        ax.axvline(0, color='#999999', lw=.6)
        for i, (term, label) in enumerate(TERMS):
            for k, (opt, col, off) in enumerate([('adam', '#222222', -.15), ('rms', '#1f77b4', .15)]):
                r = fb[(fb.task == task) & (fb.optimizer == opt) & (fb.contrast == term)]
                if not len(r):
                    continue
                r = r.iloc[0]
                ax.errorbar(r.diff_pp, i + off, xerr=[[r.diff_pp - r.ci95_low], [r.ci95_high - r.diff_pp]],
                            fmt='o', ms=3, color=col, lw=.8, label=('Adam' if opt == 'adam' else 'RMS') if i == 0 else None)
                if r.get('p_holm') is not None and r.p_holm < .05:
                    ax.text(r.ci95_high + .6, i + off, '*', va='center', fontsize=7, color=col)
        ax.set_yticks(range(len(TERMS)), [t[1] for t in TERMS] if c == 0 else [''] * len(TERMS))
        ax.invert_yaxis()
        ax.set_xlabel('effect on unseen-delay conflict acc. (pp)')
        ax.set_title(TASK_LABELS[task])
        if c == 0:
            ax.legend(frameon=False, loc='lower left')
        panel_label(ax, 'abc'[c])
    for c, task in enumerate(['early', 'late', 'dms']):
        ax = fig.add_subplot(gs[1, c])
        frames = [pd.read_csv(a.root / f'{a.prefix}_{task}_adam' / 'metrics.csv'),
                  pd.read_csv(a.root / f'{a.prefix}C_{task}_adam' / 'metrics.csv')]
        m = pd.concat(frames)
        m = m[m.split == 'test_delay']
        for cond, label, col in CONTROLS:
            s = m[m.condition == cond]
            if not len(s):
                continue
            ax.errorbar(s.mean_rate_hz.mean(), 100 * s.accuracy_incongruent.mean(),
                        xerr=s.mean_rate_hz.std() / np.sqrt(len(s)), yerr=100 * s.accuracy_incongruent.std() / np.sqrt(len(s)),
                        fmt='o', ms=4, color=col, lw=.8, label=label if c == 0 else None)
        ax.axhline(50, color='#999999', lw=.6, ls='--')
        ax.set_xlabel('mean rate, unseen-delay trials (Hz)')
        if c == 0:
            ax.set_ylabel('conflict accuracy (%)')
        ax.set_title(f'{TASK_LABELS[task]}: local learner, Adam', fontsize=7)
        panel_label(ax, 'def'[c])
    fig.legend(*fig.axes[3].get_legend_handles_labels(), loc='lower center', ncol=6, frameon=False,
               bbox_to_anchor=(.5, -.02), fontsize=6)
    a.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out / f'{a.tag}.pdf', bbox_inches='tight')
    fig.savefig(a.out / f'{a.tag}.png', dpi=200, bbox_inches='tight')
    print('saved', a.out / f'{a.tag}.pdf')


if __name__ == '__main__':
    main()
