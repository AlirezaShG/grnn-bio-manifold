"""Run the v12 confirmation as a job queue over GPU and CPU slots.

Every job of one seed runs on the same device type, so paired comparisons
never mix GPU and CPU arithmetic. Longest jobs (change objective) start first.
python scripts/v12_queue.py --out results/v12/conf --seeds 1001-1010 --gpu_seeds 1001-1003 --gpu_slots 6 --cpu_slots 16
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def rng(s):
    a, b = s.split('-') if '-' in s else (s, s)
    return list(range(int(a), int(b) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--seeds', required=True)
    ap.add_argument('--gpu_seeds', default='')
    ap.add_argument('--gpu_slots', type=int, default=6)
    ap.add_argument('--cpu_slots', type=int, default=16)
    ap.add_argument('--cohort', default='confirmation')
    ap.add_argument('--settings', type=Path, required=True, help='json: per objective/rule args')
    a = ap.parse_args()
    settings = json.loads(a.settings.read_text())
    gpu_seeds = set(rng(a.gpu_seeds)) if a.gpu_seeds else set()
    jobs = []
    for obj in ('change', 'data'):
        for rule, extra in settings[obj].items():
            for s in rng(a.seeds):
                jobs.append((obj, rule, s, extra))
    for s in rng(a.seeds):
        jobs.append(('data', 'untrained', s, []))
    logs = Path(a.out) / 'logs'; logs.mkdir(parents=True, exist_ok=True)
    running = {'cuda': [], 'cpu': []}
    slots = {'cuda': a.gpu_slots, 'cpu': a.cpu_slots}
    pending = list(jobs)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    t0 = time.time()
    while pending or any(running.values()):
        for dev in running:
            running[dev] = [p for p in running[dev] if p.poll() is None]
        for i, job in enumerate(list(pending)):
            obj, rule, s, extra = job
            dev = 'cuda' if s in gpu_seeds else 'cpu'
            if len(running[dev]) >= slots[dev]:
                continue
            cmd = [PY, str(ROOT / 'scripts/v12_train.py'), '--objective', obj, '--rule', rule, '--seed', str(s),
                   '--cohort', a.cohort, '--device', dev, '--out', a.out] + [str(x) for x in extra]
            f = open(logs / f'{obj}_{rule}_{s}.log', 'w')
            running[dev].append(subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT)))
            pending.remove(job)
        done = len(jobs) - len(pending) - sum(len(v) for v in running.values())
        print(f'{(time.time() - t0) / 60:.1f} min: done {done}/{len(jobs)}, running gpu {len(running["cuda"])} cpu {len(running["cpu"])}', flush=True)
        time.sleep(30)
    print('all jobs finished', flush=True)


if __name__ == '__main__':
    main()
