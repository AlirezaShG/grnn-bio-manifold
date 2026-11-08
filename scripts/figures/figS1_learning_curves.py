"""Supplementary Fig. S1: validation learning curves (ID trials) per learner, task, optimizer."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import apply_style, LEARNER_COLORS, LEARNER_LABELS, TASK_LABELS  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
LEARNERS = ['bptt', 'local', 'local_no_beta', 'readout_only']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--prefix', required=True)
    ap.add_argument('--out', type=Path, default=ROOT / 'paper' / 'figures')
    a = ap.parse_args()
    apply_style()
    fig, axes = plt.subplots(2, 6, figsize=(7.2, 3.2), sharex=True)
    for c, (task, opt) in enumerate([(t, o) for o in ['adam', 'rms'] for t in ['early', 'late', 'dms']]):
        for l in LEARNERS:
            curves = []
            for p in sorted((a.root / f'{a.prefix}_{task}_{opt}' / 'runs' / f'intact_{l}').glob('*/learning_curve.csv')):
                d = pd.read_csv(p).dropna(subset=['validation_bce']).set_index('step')
                curves.append(d[['validation_accuracy', 'validation_bce']])
            if not curves:
                continue
            acc = pd.concat([x.validation_accuracy for x in curves], axis=1).mean(axis=1)
            bce = pd.concat([x.validation_bce for x in curves], axis=1).mean(axis=1)
            axes[0, c].plot(acc.index, 100 * acc, color=LEARNER_COLORS[l], lw=.9, label=LEARNER_LABELS[l])
            axes[1, c].plot(bce.index, bce, color=LEARNER_COLORS[l], lw=.9)
        axes[0, c].set_title(f'{TASK_LABELS[task]}\n{"Adam" if opt == "adam" else "RMS"}', fontsize=6.5)
        axes[0, c].set_ylim(50, 101)
        axes[1, c].set_yscale('log')
        axes[1, c].set_xlabel('update')
    axes[0, 0].set_ylabel('validation acc. (%)')
    axes[1, 0].set_ylabel('validation BCE')
    axes[0, 0].legend(frameon=False, fontsize=5)
    fig.tight_layout()
    fig.savefig(a.out / 'figS1_learning_curves.pdf', bbox_inches='tight')
    fig.savefig(a.out / 'figS1_learning_curves.png', dpi=200, bbox_inches='tight')
    print('saved')


if __name__ == '__main__':
    main()
