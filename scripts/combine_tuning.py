"""Merge stage-1 grid and stage-2 lr_out refinement into one selection per learner.

Criterion is unchanged: lowest mean best-validation BCE over the 3 development
seeds (ties -> earlier stage, lower lr_out). Writes results/v11/tune/final_selection.json
and a full table of every candidate evaluated.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TUNE = ROOT / 'results' / 'v11' / 'tune'
TUNE2 = ROOT / 'results' / 'v11' / 'tune2'
rows = []
for task in ['early', 'late', 'dms']:
    for opt in ['adam', 'rms']:
        for r in json.loads((TUNE / f't_{task}_{opt}' / 'tuning_summary.json').read_text()):
            rows.append({**r, 'task': task, 'optimizer': opt, 'stage': 1})
        for learner in ['bptt', 'local', 'local_no_beta', 'readout_only']:
            p = TUNE2 / f't2_{task}_{opt}_{learner}' / 'tuning_summary.json'
            if p.exists():
                for r in json.loads(p.read_text()):
                    rows.append({**r, 'task': task, 'optimizer': opt, 'stage': 2})
df = pd.DataFrame(rows)
final, notes = {}, []
for (task, opt, learner), g in df.groupby(['task', 'optimizer', 'learner']):
    best = g.sort_values(['mean_validation_bce', 'stage', 'lr_out']).iloc[0]
    final.setdefault(f'{task}_{opt}', {})[learner] = {'lr_rec': float(best.lr_rec), 'lr_out': float(best.lr_out)}
    if best.lr_out == df.lr_out.max():
        notes.append(f'{task}/{opt}/{learner}: selected lr_out at the largest value tried ({best.lr_out})')
(TUNE / 'final_selection.json').write_text(json.dumps(final, indent=2) + '\n')
df.to_csv(TUNE / 'all_candidates.csv', index=False)
(TUNE / 'final_selection_notes.json').write_text(json.dumps({'criterion': 'mean best validation BCE, 3 dev seeds',
    'stages': 'stage 1: 4x4 grid; stage 2: lr_out in {3,10} at each learner stage-1 lr_rec', 'edge_notes': notes}, indent=2))
for k, v in final.items():
    print(k, v)
print('\n'.join(notes) if notes else 'no selection at the upper edge')
