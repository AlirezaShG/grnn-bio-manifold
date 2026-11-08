"""Write the v11 experiment configurations (development and confirmation).

Every config is generated here, so the design lives in one reviewed file.
Run: python scripts/v11_configs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grnn_bio.config import ExperimentConfig  # noqa: E402

OUT = ROOT / 'configs' / 'v11'

TASKS = {
    'early': dict(task_type='context', context_timing='early'),
    'late': dict(task_type='context', context_timing='late'),
    'dms': dict(task_type='dms', context_timing='late'),
}
OPTIMIZERS = {
    # Adam with global-norm clipping: standard, but neither momentum nor the
    # global clip is synapse-local.
    'adam': dict(optimizer='adam', max_grad_norm=1.0),
    # Per-synapse RMS normalisation, no momentum, no global clipping.
    'rms': dict(optimizer='rmsprop', max_grad_norm=None),
}
CORE_LEARNERS = ['bptt', 'local', 'local_no_beta', 'readout_only']


def base(name, *, stage='development', task='early', optimizer='adam', seeds=(11, 12), steps=2000,
         conditions=(), learning_rates=None, checkpoints=None, **extra):
    cfg = dict(
        name=name, stage=stage, cell_cohort=stage, cohort_unit='donor', holdout_fraction=0.5,
        split_seed=1701, seeds=list(seeds), cell_counts=[80, 10, 5, 5],
        task=dict(**TASKS[task]), conditions=list(conditions), training_steps=steps, batch_size=32,
        validation_interval=50, validation_trials=256, test_trials=512, analysis_trials=128,
        checkpoint_steps=sorted(set(checkpoints or [0, steps // 8, steps // 4, steps // 2, steps])),
        lr_rec=0.001, lr_out=0.03, learning_rates=learning_rates or {}, max_weight=2.0,
        save_trajectories=False, run_interventions=True, run_psychometrics=True, run_fixed_points=False,
        **OPTIMIZERS[optimizer],
    )
    cfg.update(extra)
    ExperimentConfig.from_dict(json.loads(json.dumps(cfg)))  # validate
    return cfg


def cond(name, learner, identity='authentic', topology='type_prior', calibration='matched', **kw):
    return dict(name=name, learner=learner, identity=identity, topology=topology,
                calibration=calibration, feedback_mode='random', training_noise_std=0.0, **kw)


def core_conditions():
    return [cond('intact_' + l, l) for l in CORE_LEARNERS]


def write(cfg, sub):
    path = OUT / sub / f"{cfg['name']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2) + '\n', encoding='utf-8')
    return path


def budget_pilot():
    """D0: how many updates until validation loss stops improving? (dev cohort)."""
    paths = []
    for task in TASKS:
        cfg = base(f'd0_budget_{task}', task=task, seeds=(1, 2), steps=3000,
                   conditions=core_conditions(), checkpoints=[0, 500, 1000, 2000, 3000],
                   run_interventions=False, run_psychometrics=False)
        paths.append(write(cfg, 'dev'))
    return paths


GRID = [dict(lr_rec=r, lr_out=o) for r in (3e-4, 1e-3, 3e-3, 1e-2) for o in (0.03, 0.1, 0.3, 1.0)]
STEPS = 3000


def tuning():
    """D1: validation-only learning-rate selection per learner, task and optimizer."""
    paths = []
    for task in TASKS:
        for opt in OPTIMIZERS:
            cfg = base(f't_{task}_{opt}', task=task, optimizer=opt, seeds=(11, 12, 13), steps=STEPS,
                       conditions=core_conditions(), checkpoints=[0, STEPS],
                       tuning_candidates=GRID, run_interventions=False, run_psychometrics=False)
            paths.append(write(cfg, 'tune'))
    return paths


def selected_rates(task, opt):
    """Final rates from the two-stage validation-only search (stage 2 refines lr_out)."""
    final = ROOT / 'results' / 'v11' / 'tune' / 'final_selection.json'
    if not final.exists():
        return None
    return json.loads(final.read_text())[f'{task}_{opt}']


def stage1_rates(task, opt):
    rec = ROOT / 'results' / 'v11' / 'tune' / f't_{task}_{opt}' / 'selection_record.json'
    return json.loads(rec.read_text())['selected_learning_rates']


def tuning_stage2():
    """D1b: at each learner's stage-1 lr_rec, extend lr_out upward (3, 10)."""
    paths = []
    for task in TASKS:
        for opt in OPTIMIZERS:
            sel = stage1_rates(task, opt)
            for learner in CORE_LEARNERS:
                rec = sel[learner]['lr_rec']
                cfg = base(f't2_{task}_{opt}_{learner}', task=task, optimizer=opt, seeds=(11, 12, 13), steps=STEPS,
                           conditions=[cond('intact_' + learner, learner)], checkpoints=[0, STEPS],
                           tuning_candidates=[dict(lr_rec=rec, lr_out=3.0), dict(lr_rec=rec, lr_out=10.0)],
                           run_interventions=False, run_psychometrics=False)
                paths.append(write(cfg, 'tune2'))
    return paths


def family_ab():
    return core_conditions() + [
        cond('shuffle_bptt', 'bptt', 'within_type_shuffle'),
        cond('shuffle_local', 'local', 'within_type_shuffle'),
        cond('rewired_bptt', 'bptt', topology='degree_rewired'),
        cond('rewired_local', 'local', topology='degree_rewired'),
        cond('shuffle_rewired_bptt', 'bptt', 'within_type_shuffle', 'degree_rewired'),
        cond('shuffle_rewired_local', 'local', 'within_type_shuffle', 'degree_rewired'),
        cond('homog_bptt', 'bptt', 'type_homogeneous'),
        cond('homog_local', 'local', 'type_homogeneous'),
    ]


def family_c():
    sym = dict(feedback_mode='symmetric')
    out = [
        cond('shuffle_ref_local', 'local', 'within_type_shuffle', calibration='reference'),
        cond('fastdecay_local', 'local', 'fast_decay'),
        cond('crosstype_local', 'local', 'cross_type_permutation'),
        cond('intact_local_sym', 'local'),
        cond('intact_no_beta_sym', 'local_no_beta'),
    ]
    for c in out[3:]:
        c.update(sym)
    return out


OP_FRACTIONS = (0.02, 0.1, 0.2)   # 0.05 is the intact condition of the main suites


def family_d():
    out = []
    for frac in OP_FRACTIONS:
        for learner in ('bptt', 'local'):
            out.append(cond(f'op{int(round(frac * 100)):02d}_{learner}', learner, baseline_fraction=frac))
    return out


def full_design(stage, seeds, prefix, sub):
    """Families A+B per task x optimizer, family C per task (Adam)."""
    paths = []
    for task in TASKS:
        for opt in OPTIMIZERS:
            rates = selected_rates(task, opt)
            if rates is None:
                raise SystemExit(f'missing tuning selection for {task}/{opt}')
            cfg = base(f'{prefix}_{task}_{opt}', stage=stage, task=task, optimizer=opt, seeds=seeds,
                       steps=STEPS, conditions=family_ab(), learning_rates=rates)
            paths.append(write(cfg, sub))
        rates = selected_rates(task, 'adam')
        cfg = base(f'{prefix}C_{task}_adam', stage=stage, task=task, optimizer='adam', seeds=seeds,
                   steps=STEPS, conditions=family_c(),
                   learning_rates={k: v for k, v in rates.items() if k in ('local', 'local_no_beta')})
        paths.append(write(cfg, sub))
        cfg = base(f'{prefix}D_{task}_adam', stage=stage, task=task, optimizer='adam', seeds=seeds,
                   steps=STEPS, conditions=family_d(),
                   learning_rates={k: v for k, v in rates.items() if k in ('bptt', 'local')})
        paths.append(write(cfg, sub))
    return paths


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['tuning', 'tuning2', 'rehearsal', 'confirmation'])
    a = ap.parse_args()
    if a.stage == 'tuning':
        made = budget_pilot() + tuning()
    elif a.stage == 'tuning2':
        made = tuning_stage2()
    elif a.stage == 'rehearsal':
        made = full_design('development', (21, 22, 23, 24), 'r', 'rehearsal')
    else:
        made = full_design('confirmation', tuple(range(1001, 1017)), 'c', 'confirmation')
    for p in made:
        print(p.relative_to(ROOT))
