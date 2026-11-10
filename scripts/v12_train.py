"""Train one V1 circuit with one learning rule on one objective, then evaluate it.

python scripts/v12_train.py --objective data --rule local --seed 1 --cohort development --out results/v12/dev
Writes <out>/<objective>/<rule>/<seed>/: config.json, history.csv, checkpoints.npz,
eval_<tag>.npz (natural-scene responses and grating statistics) for the
untrained network, the selected checkpoint and the performance-matched checkpoints.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grnn_bio.data import load_dataset, partition_cells  # noqa: E402
from grnn_v1.core import Circuit, CircuitConfig, LGN, load_natural_scenes, load_movie_frames  # noqa: E402
from grnn_v1.learn import DataObjective, ChangeObjective, compute_gradients, ExpAdam  # noqa: E402
from grnn_v1 import evaluate  # noqa: E402


def build(args, device):
    df = partition_cells(load_dataset(ROOT / 'model/gfr_dataset.json', ROOT / 'data/metadata.csv'),
                         args.cohort, holdout_fraction=.5, seed=1701, unit='donor')
    cfg = CircuitConfig(lgn_gain=args.lgn_gain, rec_gain=args.rec_gain, inh_gain=args.inh_gain)
    c = Circuit(df, cfg, args.seed, device)
    lgn = LGN(device)
    c.build_lgn_weights(lgn)
    c.set_ff_baseline(lgn(torch.zeros(1, 5, 48, 64, device=device)))
    return c, lgn, cfg


def state(c):
    return {'mag': c.mag.detach().cpu().numpy().astype(np.float32), 'w_out': c.w_out.detach().cpu().numpy(),
            'b_out': c.b_out.detach().cpu().numpy()}


def load_state(c, s):
    with torch.no_grad():
        c.mag.copy_(torch.tensor(s['mag'], device=c.device))
        c.w_out.copy_(torch.tensor(s['w_out'], device=c.device))
        c.b_out.copy_(torch.tensor(s['b_out'], device=c.device))


def run_eval(c, lgn, scenes, seed, path, extra=None):
    ns = evaluate.natural_scenes(c, lgn, scenes, n_rep=10, seed=seed)
    gr = evaluate.gratings(c, lgn, n_rep=5, seed=seed)
    np.savez_compressed(path, ns=ns.astype(np.float16), types=c.types, theta=c.theta,
                        **{f'gr_{k}': v for k, v in gr.items()}, **(extra or {}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--objective', choices=['data', 'change'], required=True)
    ap.add_argument('--rule', choices=['bptt', 'local', 'local_nobeta', 'readout', 'local_sym', 'untrained'], required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--cohort', default='development')
    ap.add_argument('--steps', type=int, default=1000)
    ap.add_argument('--lr_rec', type=float, default=5e-3)
    ap.add_argument('--lr_out', type=float, default=1e-2)
    ap.add_argument('--lgn_gain', type=float, default=0.2)
    ap.add_argument('--rec_gain', type=float, default=1.0)
    ap.add_argument('--inh_gain', type=float, default=2.0)
    ap.add_argument('--val_every', type=int, default=50)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--no_eval', action='store_true')
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    device = torch.device(a.device)
    torch.set_num_threads(1 if a.device == 'cpu' else torch.get_num_threads())
    torch.manual_seed(a.seed)
    out = a.out / a.objective / a.rule / str(a.seed)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'done.json').exists():
        print('already done', out); return
    (out / 'config.json').write_text(json.dumps(vars(a), default=str, indent=2))
    c, lgn, cfg = build(a, device)
    scenes = load_natural_scenes()
    if a.objective == 'data':
        obj = DataObjective(c, lgn, device)
        val_batch = obj.batch(np.random.default_rng(10_000 + a.seed))
        better = lambda new, old: new < old
    else:
        # Visual Behavior structure: a fixed set of 8 images, chosen as the 8 most
        # mutually distinct movie frames (greedy farthest-point). The 118 test scenes
        # are never used for training.
        allf = load_movie_frames(step=5)
        flat = allf.reshape(len(allf), -1)
        chosen = [0]
        for _ in range(7):
            d = np.min(((flat[:, None, :] - flat[None, chosen, :]) ** 2).sum(-1), axis=1)
            chosen.append(int(d.argmax()))
        frames = allf[chosen]
        train_ids = val_ids = np.arange(8)
        obj = ChangeObjective(c, lgn, device, frames)
        val_batches = [obj.batch(np.random.default_rng(20_000 + a.seed + k), val_ids) for k in range(4)]
        better = lambda new, old: new > old
    rule = 'local' if a.rule == 'local_sym' else a.rule
    feedback = 'symmetric' if a.rule == 'local_sym' else 'random'
    opt = ExpAdam(c, a.lr_rec, a.lr_out)

    def validate():
        with torch.no_grad():
            if a.objective == 'data':
                r, y = c.run(val_batch['lgn'], noise_seed=999)
                return obj.metric(r, y, val_batch)
            accs = []
            for k, vb in enumerate(val_batches):
                r, y = c.run(vb['lgn'], noise_seed=900 + k)
                accs.append(obj.metric(r, y, vb))
            return float(np.mean(accs))

    hist, ckpts = [], {}
    v0 = validate(); best, best_step = v0, 0
    ckpts['step_0'] = state(c)
    hist.append({'step': 0, 'val': v0})
    t0 = time.time()
    steps = 0 if a.rule == 'untrained' else a.steps
    for step in range(1, steps + 1):
        rng = np.random.default_rng(a.seed * 100_000 + step)
        batch = obj.batch(rng) if a.objective == 'data' else obj.batch(rng, train_ids)
        L, rates, y = compute_gradients(c, obj, batch, rule, noise_seed=step, feedback=feedback)
        if a.objective == 'change' and a.rule != 'readout':
            pass
        opt.step()
        row = {'step': step, 'loss': float(L['total']), 'task': float(L['task']), 'direct': float(L['direct']),
               'mean_rate': float(rates.mean())}
        if step % a.val_every == 0 or step == steps:
            v = validate(); row['val'] = v
            if better(v, best):
                best, best_step = v, step
                ckpts['best'] = state(c)
            if step % (a.val_every * 4) == 0 or step == steps:
                ckpts[f'step_{step}'] = state(c)
            print(f'{a.objective}/{a.rule}/{a.seed} step {step} loss {row["loss"]:.4f} val {v:.4f} '
                  f'rate {row["mean_rate"]:.1f} ({time.time() - t0:.0f}s)', flush=True)
        if not np.isfinite(row['loss']):
            print('nonfinite loss; stopping', flush=True); break
        hist.append(row)
    if 'best' not in ckpts:
        ckpts['best'] = state(c) if a.rule != 'untrained' else ckpts['step_0']
    pd.DataFrame(hist).to_csv(out / 'history.csv', index=False)
    np.savez_compressed(out / 'checkpoints.npz', **{f'{k}__{n}': v for k, s in ckpts.items() for n, v in s.items()})
    summary = {'best_val': best, 'best_step': best_step, 'initial_val': v0, 'train_seconds': time.time() - t0}
    if not a.no_eval:
        load_state(c, ckpts['best'])
        run_eval(c, lgn, scenes, a.seed, out / 'eval_best.npz')
        if a.objective == 'change':
            summary['val_acc_best'] = validate()
        # Geometry along training: natural-scene responses at every saved checkpoint,
        # with the validation value at that step, to compare rules at matched performance.
        traj, vals = {}, {}
        hv = {int(r['step']): r.get('val') for r in hist if r.get('val') is not None}
        for k in sorted(kk for kk in ckpts if kk.startswith('step_')):
            step = int(k.split('_')[1])
            load_state(c, ckpts[k])
            traj[k] = evaluate.natural_scenes(c, lgn, scenes, n_rep=10, seed=a.seed).astype(np.float16)
            vals[k] = hv.get(step, np.nan)
        np.savez_compressed(out / 'eval_traj.npz', types=c.types, steps=np.array(sorted(traj, key=lambda s: int(s.split('_')[1]))),
                            vals=np.array([vals[k] for k in sorted(traj, key=lambda s: int(s.split('_')[1]))]), **traj)
    (out / 'done.json').write_text(json.dumps(summary, indent=2))
    print('done', summary, flush=True)


if __name__ == '__main__':
    main()
