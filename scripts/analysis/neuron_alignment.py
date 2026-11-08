"""Per-neuron credit-assignment alignment versus each cell's spike-history feedback.

For each finished run, rebuild the circuit, load the selected checkpoint, and on
the run's own independent diagnostic batch compute, for every postsynaptic
neuron i, the cosine between row i of the exact recurrent gradient and row i of
the local estimate (symmetric feedback, so only the truncation of paths through
other neurons remains), with and without the beta term.

Cell property: adaptation loop gain = |g'_i * mean_k(beta_ik/ds_k)| at the
operating point (mean slope over the diagnostic trials), dimensionless.

Usage: python scripts/analysis/neuron_alignment.py <suite dir> [<suite dir> ...] --out file.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for var in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(var, '1')

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from grnn_bio.config import ExperimentConfig  # noqa: E402
from grnn_bio.data import load_dataset, partition_cells  # noqa: E402
from grnn_bio.experiment import build_circuit  # noqa: E402
from grnn_bio.randomness import rng_for  # noqa: E402
from grnn_bio.task import generate_batch  # noqa: E402
from grnn_bio.learners import exact_bptt_gradients, local_eligibility_gradients, _forward_cache  # noqa: E402


def _row_cos(a, b):
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    out = np.full(len(a), np.nan)
    ok = den > 1e-18
    out[ok] = num[ok] / den[ok]
    return out


def analyse_run(args):
    suite, condition_name, seed = args
    cfg = ExperimentConfig.load(Path(suite) / 'config.json')
    condition = next(c for c in cfg.conditions if c.name == condition_name)
    full = load_dataset(ROOT / 'model/gfr_dataset.json', ROOT / 'data/metadata.csv', cfg.min_validation_evr)
    cohort = partition_cells(full, cfg.cell_cohort, holdout_fraction=cfg.holdout_fraction,
                             seed=cfg.split_seed, unit=cfg.cohort_unit)
    circuit, rows = build_circuit(cfg, cohort, condition, seed)
    run_dir = Path(suite) / 'runs' / condition_name / str(seed)
    with np.load(run_dir / 'best_checkpoint.npz', allow_pickle=False) as data:
        circuit.load_state_dict({k[len('model__'):]: data[k] for k in data.files if k.startswith('model__')})
    batch = generate_batch(cfg.task, min(32, cfg.analysis_trials), rng_for(seed, cfg.stage + '/gradient_diagnostic'))
    exact = exact_bptt_gradients(circuit, batch)['rec']
    est = {f'{"beta" if b else "nobeta"}_{m}': local_eligibility_gradients(circuit, batch, include_beta=b, feedback_mode=m)['rec']
           for b in (True, False) for m in ('symmetric', 'random')}
    _, slopes, _ = _forward_cache(circuit, np.asarray(batch.X, float))
    slope = slopes.mean(axis=(0, 1))  # Hz/pA, mean over trials and time
    hist_gain = (circuit.beta / circuit.ds).mean(axis=1)  # pA/Hz
    frame = pd.DataFrame({
        'suite': Path(suite).name, 'condition': condition_name, 'learner': condition.learner,
        'identity': condition.identity, 'seed': seed, 'slot': np.arange(circuit.N),
        'cell_type': circuit.cell_types, 'in_degree': circuit.mask.sum(1),
        'history_gain_pA_per_Hz': hist_gain, 'mean_slope_Hz_per_pA': slope,
        'adaptation_loop_gain': np.abs(slope * hist_gain),
        'exact_row_norm': np.linalg.norm(exact, axis=1),
    })
    for key, g in est.items():
        frame[f'cos_{key}'] = _row_cos(g, exact)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('suites', nargs='+', type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--conditions', nargs='*', default=None)
    a = ap.parse_args()
    tasks = []
    for suite in a.suites:
        cfg = json.loads((suite / 'config.json').read_text())
        for c in cfg['conditions']:
            if a.conditions and c['name'] not in a.conditions:
                continue
            for s in cfg['seeds']:
                if (suite / 'runs' / c['name'] / str(s) / 'completed.json').exists():
                    tasks.append((str(suite), c['name'], s))
    with ProcessPoolExecutor(a.jobs) as pool:
        frames = list(pool.map(analyse_run, tasks))
    out = pd.concat(frames, ignore_index=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f'{len(tasks)} runs -> {a.out}')
    summary = out.groupby(['suite', 'condition', 'cell_type'])[[c for c in out if c.startswith('cos_')]].mean().round(3)
    print(summary.to_string())


if __name__ == '__main__':
    main()
