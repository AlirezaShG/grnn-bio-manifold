"""v12 (GPU): local e-prop gradient correctness and basic circuit behaviour."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grnn_bio.data import load_dataset, partition_cells  # noqa: E402
from grnn_v1.core import Circuit, CircuitConfig, LGN  # noqa: E402
from grnn_v1.learn import DataObjective, ChangeObjective, compute_gradients  # noqa: E402

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def small(seed=3):
    df = partition_cells(load_dataset(ROOT / 'model/gfr_dataset.json', ROOT / 'data/metadata.csv'),
                         'development', holdout_fraction=.5, seed=1701, unit='donor')
    cfg = CircuitConfig(counts={'Pyr': 48, 'PV': 8, 'SST': 4, 'VIP': 4})
    c = Circuit(df, cfg, seed, DEV)
    lgn = LGN(DEV)
    c.build_lgn_weights(lgn)
    c.set_ff_baseline(lgn(torch.zeros(1, 5, 48, 64, device=DEV)))
    return c, lgn


class LocalGradientTest(unittest.TestCase):
    def test_local_equals_bptt_without_network_paths(self):
        c, lgn = small()
        with torch.no_grad():
            c.w_out.normal_(0, .5)
            saved = c.mag.detach().clone()
        obj = ChangeObjective(c, lgn, DEV, frames=np.random.default_rng(0).uniform(-1, 1, (60, 48, 64)).astype(np.float32), batch_size=8)
        batch = obj.batch(np.random.default_rng(1))
        # W = 0 numerically, while d|W| still matters: no path through other cells exists.
        with torch.no_grad():
            c.mag.mul_(0.0)
        compute_gradients(c, obj, batch, 'bptt', noise_seed=5)
        g_bptt = c.mag.grad.clone()
        compute_gradients(c, obj, batch, 'local', noise_seed=5, feedback='symmetric')
        g_loc = c.mag.grad.clone()
        rel = (g_loc - g_bptt).norm() / g_bptt.norm()
        self.assertGreater(float(g_bptt.norm()), 0)
        self.assertLess(float(rel), 1e-4)
        with torch.no_grad():
            c.mag.copy_(saved)

    def test_local_differs_with_network_paths_and_beta_matters(self):
        c, lgn = small()
        obj = DataObjective(c, lgn, DEV, n_gray=2, n_rep=1)
        batch = obj.batch(np.random.default_rng(2))
        compute_gradients(c, obj, batch, 'bptt', noise_seed=1); g_b = c.mag.grad.clone()
        compute_gradients(c, obj, batch, 'local', noise_seed=1); g_l = c.mag.grad.clone()
        compute_gradients(c, obj, batch, 'local_nobeta', noise_seed=1); g_n = c.mag.grad.clone()
        cos = lambda a, b: float((a * b).sum() / (a.norm() * b.norm()))
        self.assertLess(cos(g_l, g_b), 0.9999)
        self.assertGreater(float((g_l - g_n).norm() / g_l.norm()), 1e-3)

    def test_bptt_matches_finite_difference(self):
        c, lgn = small()
        obj = DataObjective(c, lgn, DEV, n_gray=2, n_rep=1)
        batch = obj.batch(np.random.default_rng(4))
        compute_gradients(c, obj, batch, 'bptt', noise_seed=2)
        g = c.mag.grad.clone()
        idx = torch.nonzero(c.mask)[torch.argmax(g[c.mask.bool()].abs())]
        i, j = int(idx[0]), int(idx[1])
        eps = 1e-3 * float(c.mag[i, j])
        with torch.no_grad():
            c.mag[i, j] += eps
            lp = float(obj.loss(*c.run(batch['lgn'], noise_seed=2), batch)['total'])
            c.mag[i, j] -= 2 * eps
            lm = float(obj.loss(*c.run(batch['lgn'], noise_seed=2), batch)['total'])
            c.mag[i, j] += eps
        fd = (lp - lm) / (2 * eps)
        self.assertLess(abs(fd - float(g[i, j])) / (abs(fd) + 1e-12), 2e-2)


if __name__ == '__main__':
    unittest.main()
