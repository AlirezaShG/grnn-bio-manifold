"""Run several validation-only tuning searches through ONE process pool.

Usage: python scripts/tune_many.py --jobs 26 --out results/v11/tune configs/v11/tune/t_*.json
"""
import argparse, os, sys, time
from pathlib import Path
for var in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(var, '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grnn_bio.config import ExperimentConfig
from grnn_bio.experiment import execute_jobs
from grnn_bio.tuning import plan_tuning, finish_tuning


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('configs', nargs='+', type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--resume', action='store_true')
    a = ap.parse_args()
    plans, jobs = [], []
    for path in a.configs:
        cfg = ExperimentConfig.load(path)
        if cfg.stage != 'development' or cfg.cell_cohort != 'development':
            raise SystemExit('tuning is development-only')
        out = a.out / cfg.name
        suites, js = plan_tuning(cfg, out, ROOT, a.resume)
        plans.append((cfg, out, suites, len(jobs), len(js)))
        jobs.extend(js)
    order = sorted(range(len(jobs)), key=lambda i: jobs[i][2].learner not in ('local', 'local_no_beta'))
    t0 = time.time()
    print(f'{len(jobs)} tuning runs from {len(plans)} searches on {a.jobs} workers', flush=True)
    done = execute_jobs([jobs[i] for i in order], a.jobs)
    results = [None] * len(jobs)
    for k, i in enumerate(order):
        results[i] = done[k]
    for cfg, out, suites, off, n in plans:
        sel = finish_tuning(cfg, out, suites, results[off:off + n], ROOT)
        print(cfg.name, sel, flush=True)
    print(f'finished in {(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
