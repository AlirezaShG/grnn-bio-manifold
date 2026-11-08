"""v11: the reverse-mode local gradient must equal the forward eligibility traces."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grnn_bio.data import load_dataset, partition_cells, sample_cells
from grnn_bio.model import calibrate_current
from grnn_bio.circuit import Circuit
from grnn_bio.task import TaskConfig, generate_batch
from grnn_bio.learners import (local_eligibility_gradients, exact_bptt_gradients,
                               ProjectedOptimizer)


def _relative(a, b):
    return np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300)


class AdjointLocalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = partition_cells(load_dataset(ROOT / 'model/gfr_dataset.json', ROOT / 'data/metadata.csv'), 'development')
        params, rows = sample_cells(df, (12, 3, 2, 2), 3)
        cls.circuit = Circuit(params, rows.cell_type.to_numpy(), seed=5, calibration=calibrate_current(params))
        # Move weights away from initialization so slopes and beta paths are active.
        rng = np.random.default_rng(1)
        cls.circuit.W_abs[...] = cls.circuit.W_abs * rng.uniform(.5, 1.5, cls.circuit.W_abs.shape)
        cls.batch = generate_batch(TaskConfig(context_timing='late'), 16, np.random.default_rng(2))

    def test_adjoint_equals_forward_traces(self):
        for include_beta in (True, False):
            for feedback_mode in ('random', 'symmetric'):
                fwd = local_eligibility_gradients(self.circuit, self.batch, include_beta=include_beta,
                                                  feedback_mode=feedback_mode, implementation='forward')
                adj = local_eligibility_gradients(self.circuit, self.batch, include_beta=include_beta,
                                                  feedback_mode=feedback_mode, implementation='adjoint')
                self.assertGreater(np.linalg.norm(fwd['rec']), 0)
                self.assertLess(_relative(adj['rec'], fwd['rec']), 1e-10, (include_beta, feedback_mode))
                np.testing.assert_allclose(adj['out'], fwd['out'], rtol=1e-12, atol=1e-15)
                self.assertAlmostEqual(adj['bias'], fwd['bias'], places=12)
                self.assertAlmostEqual(adj['loss'], fwd['loss'], places=12)

    def test_beta_term_changes_the_estimate(self):
        with_beta = local_eligibility_gradients(self.circuit, self.batch)
        without = local_eligibility_gradients(self.circuit, self.batch, include_beta=False)
        self.assertGreater(_relative(with_beta['rec'], without['rec']), 1e-3)

    def test_adjoint_with_noise_matches_forward(self):
        noise = np.random.default_rng(4).normal(0, .1, (self.batch.n_trials, self.batch.n_steps, self.circuit.N))
        fwd = local_eligibility_gradients(self.circuit, self.batch, noise=noise, implementation='forward')
        adj = local_eligibility_gradients(self.circuit, self.batch, noise=noise)
        self.assertLess(_relative(adj['rec'], fwd['rec']), 1e-10)

    def test_symmetric_local_differs_from_bptt_only_by_network_paths(self):
        # With W = 0 there are no paths through other neurons, so symmetric
        # e-prop must equal the exact network gradient.
        saved = self.circuit.W_abs.copy()
        try:
            self.circuit.W_abs[...] = 0.
            exact = exact_bptt_gradients(self.circuit, self.batch)
            local = local_eligibility_gradients(self.circuit, self.batch, feedback_mode='symmetric')
            self.assertLess(_relative(local['rec'], exact['rec']), 1e-10)
        finally:
            self.circuit.W_abs[...] = saved

    def test_unknown_implementation_rejected(self):
        with self.assertRaises(ValueError):
            local_eligibility_gradients(self.circuit, self.batch, implementation='fast')


class RmsOptimizerTest(unittest.TestCase):
    def test_rmsprop_has_no_momentum(self):
        class C:  # minimal circuit stand-in
            pass
        c = C()
        c.W_abs = np.full((2, 2), .5); c.mask = np.ones((2, 2), bool)
        c.W_out = np.zeros(2); c.readout_mask = np.ones(2, bool); c.b_out = 0.
        opt = ProjectedOptimizer(kind='rmsprop', lr_rec=.1, lr_out=.1, max_grad_norm=None)
        g = {'rec': np.array([[1., -1.], [2., 0.]]), 'out': np.zeros(2), 'bias': 0.}
        opt.step(c, g)
        # First bias-corrected step: update = lr * g / |g| (sign step), zero stays zero.
        np.testing.assert_allclose(c.W_abs, [[.4, .6], [.4, .5]], atol=1e-6)
        # Second step with zero gradient must not move weights (no momentum).
        before = c.W_abs.copy()
        opt.step(c, {'rec': np.zeros((2, 2)), 'out': np.zeros(2), 'bias': 0.})
        np.testing.assert_array_equal(c.W_abs, before)
        state = opt.state_dict()
        restored = ProjectedOptimizer(kind='rmsprop')
        restored.load_state_dict(state)
        self.assertEqual(restored.kind, 'rmsprop')


if __name__ == '__main__':
    unittest.main()
