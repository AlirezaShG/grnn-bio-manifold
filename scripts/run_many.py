"""Run several suite configs through ONE process pool (development/confirmation).

Usage: python scripts/run_many.py --jobs 26 --out results/v11/dev cfg1.json cfg2.json ...
Each config writes to <out>/<config name>/. Resumable with --resume.
"""
import argparse, os, sys, time
from pathlib import Path
for var in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(var, '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grnn_bio.config import ExperimentConfig
from grnn_bio.experiment import prepare_suite, execute_jobs, finalize_suite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('configs', nargs='+', type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--resume', action='store_true')
    a = ap.parse_args()
    suites, jobs = [], []
    stages = {ExperimentConfig.load(p).stage for p in a.configs}
    if 'confirmation' in stages:
        sys.path.insert(0, str(ROOT / 'scripts'))
        from lock_protocol import verify
        locked = verify()
        print('protocol lock verified:', locked['locked_at_utc'], flush=True)
    for path in a.configs:
        cfg = ExperimentConfig.load(path)
        out = a.out / cfg.name
        js = prepare_suite(cfg, out, project_root=ROOT, resume=a.resume, test=True)
        suites.append((cfg, out, len(jobs), len(js)))
        jobs.extend(js)
    # Longest jobs (local learners) first for better packing.
    order = sorted(range(len(jobs)), key=lambda i: jobs[i][2].learner not in ('local', 'local_no_beta'))
    t0 = time.time()
    print(f'{len(jobs)} runs from {len(suites)} suites on {a.jobs} workers', flush=True)
    done = execute_jobs([jobs[i] for i in order], a.jobs)
    results = [None] * len(jobs)
    for k, i in enumerate(order):
        results[i] = done[k]
    for cfg, out, off, n in suites:
        finalize_suite(cfg, out, results[off:off + n], project_root=ROOT, test=True)
    print(f'finished in {(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
