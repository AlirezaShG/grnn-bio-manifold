"""Hypothesis tests exactly as written in docs/PREREGISTRATION_v11.md.

Usage:
  python scripts/analysis/confirmatory_tests.py --root results/v11/confirmation --prefix c --out results/v11/confirmation/analysis
(the rehearsal uses --root results/v11/rehearsal --prefix r)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from grnn_bio.statistics import paired_comparison, holm_adjust, mean_interval  # noqa: E402
from grnn_bio.design import factorial_contrasts  # noqa: E402
from grnn_bio.io import write_json  # noqa: E402

TASKS = ['early', 'late', 'dms']
OPTS = ['adam', 'rms']
PRIMARY = ('test_delay', 'accuracy_incongruent')
SECONDARY = [('test_delay', 'accuracy'), ('test_id', 'accuracy'), ('test_combined', 'accuracy'),
             ('test_combined', 'accuracy_incongruent')]
MARGIN_PP = 5.0


def load(root: Path, prefix: str):
    frames = []
    for suite in sorted(root.glob(f'{prefix}*_*')):
        if not (suite / 'metrics.csv').exists():
            continue
        m = pd.read_csv(suite / 'metrics.csv')
        name = suite.name
        parts = name.split('_')
        m['family_suite'] = parts[0][len(prefix):] or 'AB'
        m['task'], m['optimizer'] = parts[1], parts[2]
        frames.append(m)
    if not frames:
        raise SystemExit(f'no finished suites under {root} with prefix {prefix}')
    return pd.concat(frames, ignore_index=True)


def values(m, task, opt, condition, split, metric):
    s = m[(m.task == task) & (m.optimizer == opt) & (m.condition == condition) & (m.split == split)]
    s = s.sort_values('seed')
    return s.seed.to_numpy(), 100 * s[metric].to_numpy()


def contrast(m, task, opt, a, b, split, metric, label, seed=1701):
    sa, va = values(m, task, opt, a, split, metric)
    sb, vb = values(m, task, opt, b, split, metric)
    if len(sa) == 0 or not np.array_equal(sa, sb) or not np.all(np.isfinite(va)) or not np.all(np.isfinite(vb)):
        return {'task': task, 'optimizer': opt, 'contrast': label, 'a': a, 'b': b, 'split': split,
                'metric': metric, 'status': 'not_estimable', 'p_value': None}
    r = paired_comparison(va, vb, seed=seed)
    ci90 = mean_interval(va - vb, confidence=.90, seed=seed + 7)
    return {'task': task, 'optimizer': opt, 'contrast': label, 'a': a, 'b': b, 'split': split,
            'metric': metric, 'status': 'ok', 'n': len(sa), 'mean_a': float(va.mean()), 'mean_b': float(vb.mean()),
            'diff_pp': r['mean_difference_a_minus_b'], 'ci95_low': r['ci_low'], 'ci95_high': r['ci_high'],
            'ci90_low': ci90['ci_low'], 'ci90_high': ci90['ci_high'], 'dz': r['effect_dz'],
            'p_value': r['p_value'], 'wins_a': int(np.sum(va > vb)), 'ties': int(np.sum(va == vb)),
            'test': r['test_method']}


def family_a(m, split, metric):
    rows = []
    for task in TASKS:
        for opt in OPTS:
            rows.append(contrast(m, task, opt, 'intact_local', 'intact_local_no_beta', split, metric, 'A1 local - no_beta'))
            rows.append(contrast(m, task, opt, 'intact_local', 'intact_readout_only', split, metric, 'A2 local - readout'))
            rows.append(contrast(m, task, opt, 'intact_bptt', 'intact_local', split, metric, 'A3 bptt - local'))
    for r, p in zip(rows, holm_adjust([r['p_value'] for r in rows])):
        r['p_holm'] = p
        r['family'] = 'A'
        r['family_size'] = len(rows)
    return rows


FACTORIAL_MAP = {  # condition name -> (identity, topology, learner)
    'intact_bptt': ('authentic', 'type_prior', 'bptt'), 'intact_local': ('authentic', 'type_prior', 'local'),
    'shuffle_bptt': ('within_type_shuffle', 'type_prior', 'bptt'), 'shuffle_local': ('within_type_shuffle', 'type_prior', 'local'),
    'rewired_bptt': ('authentic', 'degree_rewired', 'bptt'), 'rewired_local': ('authentic', 'degree_rewired', 'local'),
    'shuffle_rewired_bptt': ('within_type_shuffle', 'degree_rewired', 'bptt'),
    'shuffle_rewired_local': ('within_type_shuffle', 'degree_rewired', 'local'),
}


def equivalence(r):
    if r.get('status') != 'ok' or r['ci90_low'] is None:
        return 'not_estimable'
    if r['p_holm'] is not None and r['p_holm'] < .05:
        return 'different'
    if -MARGIN_PP < r['ci90_low'] and r['ci90_high'] < MARGIN_PP:
        return 'practically_negligible'
    return 'inconclusive'


def family_b(m, split, metric):
    out = []
    for task in TASKS:
        for opt in OPTS:
            sub = m[(m.task == task) & (m.optimizer == opt) & (m.split == split) & (m.family_suite == 'AB')]
            rows = []
            for _, r in sub.iterrows():
                if r.condition in FACTORIAL_MAP:
                    i, t, l = FACTORIAL_MAP[r.condition]
                    rows.append({'split': split, 'seed': int(r.seed), 'identity': i, 'topology': t,
                                 'learner': l, metric: 100 * r[metric], 'training_noise_std': 0.0})
            fac = factorial_contrasts(rows, metric=metric)
            fam = []
            for c in fac.get('contrasts', []):
                vals = np.asarray(c['per_network_contrast'])
                ci90 = mean_interval(vals, confidence=.90, seed=99)
                fam.append({'task': task, 'optimizer': opt, 'contrast': 'B factorial: ' + c['term'],
                            'split': split, 'metric': metric, 'status': 'ok', 'n': c['n_pairs'],
                            'diff_pp': c['mean_difference_a_minus_b'], 'ci95_low': c['ci_low'],
                            'ci95_high': c['ci_high'], 'ci90_low': ci90['ci_low'], 'ci90_high': ci90['ci_high'],
                            'dz': c['effect_dz'], 'p_value': c['p_value'], 'test': c['test_method'],
                            'coding': 'positive = intact identity / type-prior topology / local learner'})
            if fac.get('status') != 'complete':
                fam.append({'task': task, 'optimizer': opt, 'contrast': 'B factorial', 'status': fac.get('status'),
                            'p_value': None})
            fam.append(contrast(m, task, opt, 'intact_bptt', 'homog_bptt', split, metric, 'B homogeneity (bptt): intact - homog'))
            fam.append(contrast(m, task, opt, 'intact_local', 'homog_local', split, metric, 'B homogeneity (local): intact - homog'))
            # Simple effects for the local learner (descriptive, inside the same table).
            for r, p in zip(fam, holm_adjust([r['p_value'] for r in fam])):
                r['p_holm'] = p
                r['family'] = f'B/{task}/{opt}'
                r['family_size'] = len(fam)
                r['equivalence_5pp'] = equivalence(r)
            out.extend(fam)
            for a, b, label in [('intact_local', 'shuffle_local', 'simple: identity (local, type prior)'),
                                ('intact_bptt', 'shuffle_bptt', 'simple: identity (bptt, type prior)'),
                                ('intact_local', 'rewired_local', 'simple: topology (local, intact)'),
                                ('intact_bptt', 'rewired_bptt', 'simple: topology (bptt, intact)')]:
                r = contrast(m, task, opt, a, b, split, metric, label)
                r['family'] = 'descriptive'
                r['p_holm'] = None
                out.append(r)
    return out


def family_c(m, split, metric):
    """Cross-suite pairs (same seeds => same cells, weights, trials)."""
    rows = []
    for task in TASKS:
        mm = m[(m.task == task) & (m.optimizer == 'adam')]
        pairs = [('shuffle_ref_local', 'shuffle_local', 'C operating point: shuffle reference - matched'),
                 ('shuffle_ref_local', 'intact_local', 'C shuffle reference - intact'),
                 ('fastdecay_local', 'intact_local', 'C fast decay - intact'),
                 ('crosstype_local', 'intact_local', 'C cross-type - intact'),
                 ('intact_local_sym', 'intact_local', 'C symmetric - random feedback (local)'),
                 ('intact_local_sym', 'intact_no_beta_sym', 'C symmetric: local - no_beta'),
                 ('intact_bptt', 'intact_local_sym', 'C bptt - symmetric local')]
        for a, b, label in pairs:
            r = contrast(mm, task, 'adam', a, b, split, metric, label)
            r['family'] = 'C (descriptive)'
            r['p_holm'] = None
            rows.append(r)
    return rows


OP_LEVELS = [(0.02, 'op02'), (0.05, 'intact'), (0.10, 'op10'), (0.20, 'op20')]


def family_d(m, split, metric):
    """Operating point: per-network least-squares slope of the endpoint vs log2(fraction).

    Descriptive family (no Holm); exact sign-flip test on the per-network slopes,
    plus each level against the 5% default.
    """
    rows = []
    x = np.log2([f for f, _ in OP_LEVELS])
    for task in TASKS:
        for learner in ('bptt', 'local'):
            names = [(f'{p}_{learner}' if p != 'intact' else f'intact_{learner}') for _, p in OP_LEVELS]
            series = [values(m, task, 'adam', n, split, metric) for n in names]
            seeds = [s for s, _ in series]
            if any(len(s) == 0 for s in seeds) or any(not np.array_equal(seeds[0], s) for s in seeds):
                rows.append({'task': task, 'learner': learner, 'contrast': 'D slope', 'status': 'not_estimable', 'p_value': None})
                continue
            Y = np.stack([v for _, v in series], axis=1)
            slopes = np.polyfit(x, Y.T, 1)[0]
            r = paired_comparison(slopes, np.zeros_like(slopes), seed=31)
            rows.append({'task': task, 'optimizer': 'adam', 'learner': learner, 'family': 'D (descriptive)',
                         'contrast': f'D slope per doubling of baseline ({learner})', 'status': 'ok', 'n': len(slopes),
                         'diff_pp': r['mean_difference_a_minus_b'], 'ci95_low': r['ci_low'], 'ci95_high': r['ci_high'],
                         'p_value': r['p_value'], 'wins_a': int(np.sum(slopes > 0)), 'split': split, 'metric': metric,
                         'level_means': {f'{f:g}': float(Y[:, i].mean()) for i, (f, _) in enumerate(OP_LEVELS)}})
            for (f, p), name in zip(OP_LEVELS, names):
                if p == 'intact':
                    continue
                rr = contrast(m, task, 'adam', name, f'intact_{learner}', split, metric, f'D baseline {f:g} - 0.05 ({learner})')
                rr['family'] = 'D (descriptive)'
                rr['p_holm'] = None
                rows.append(rr)
    return rows


def condition_table(m):
    rows = []
    for (task, opt, cond), g in m.groupby(['task', 'optimizer', 'condition']):
        r = {'task': task, 'optimizer': opt, 'condition': cond, 'n': g.seed.nunique()}
        for split, metric in [PRIMARY] + SECONDARY:
            v = 100 * g[g.split == split][metric].to_numpy()
            ci = mean_interval(v, seed=5)
            r[f'{split}/{metric}'] = ci['mean']
            r[f'{split}/{metric}/ci'] = [ci['ci_low'], ci['ci_high']]
        d = g[g.split == 'test_delay']
        r['rate_hz'] = d.mean_rate_hz.mean()
        r['ceiling_pct'] = 100 * d.ceiling_rate_fraction.mean()
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, type=Path)
    ap.add_argument('--prefix', required=True)
    ap.add_argument('--out', required=True, type=Path)
    a = ap.parse_args()
    m = load(a.root, a.prefix)
    result = {'primary_endpoint': '/'.join(PRIMARY), 'equivalence_margin_pp': MARGIN_PP,
              'family_A': family_a(m, *PRIMARY), 'family_B': family_b(m, *PRIMARY),
              'family_C': family_c(m, *PRIMARY), 'family_D': family_d(m, *PRIMARY), 'conditions': condition_table(m),
              'secondary': {f'{s}/{k}': {'A': family_a(m, s, k), 'C': family_c(m, s, k)} for s, k in SECONDARY}}
    write_json(a.out / 'confirmatory_tests.json', result)
    flat = pd.DataFrame(result['family_A'] + result['family_B'] + result['family_C'] + result['family_D'])
    flat.to_csv(a.out / 'confirmatory_tests.csv', index=False)
    pd.DataFrame(result['conditions']).to_csv(a.out / 'condition_means.csv', index=False)
    show = ['family', 'task', 'optimizer', 'contrast', 'diff_pp', 'ci95_low', 'ci95_high', 'p_value', 'p_holm',
            'wins_a', 'equivalence_5pp']
    pd.set_option('display.width', 250)
    print(flat[[c for c in show if c in flat]].round(4).to_string(index=False))


if __name__ == '__main__':
    main()
