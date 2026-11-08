"""Independent mathematical and integration checks for the learning rules.

Run without pytest: python -m unittest discover -s tests -p test_learning.py -v
"""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from grnn_bio.circuit import Circuit
from grnn_bio.data import load_dataset, sample_cells
from grnn_bio.model import GFRParameters
from grnn_bio.learners import (
    ProjectedOptimizer, _forward_cache, exact_bptt_gradients,
    finite_difference_audit, local_eligibility_gradients,
    node_perturbation_gradients, readout_only_gradients,
)


ROOT = Path(__file__).resolve().parents[1]


def toy_circuit(beta=True):
    n, k = 5, 3
    a = np.tile([0.50, 0.10, -0.025], (n, 1)) * np.linspace(.8, 1.2, n)[:, None]
    b = np.tile([-.010, -.002, .0005], (n, 1)) if beta else np.zeros((n, k))
    params = GFRParameters(
        a=a, beta=b, ds=np.tile([1.0, .3, .08], (n, 1)),
        max_current=np.linspace(7., 10., n), gamma_hz=np.linspace(60., 100., n),
        poly_coeff=np.tile([.35, 1.2, .06], (n, 1)), threshold=np.zeros(n),
        cell_ids=np.arange(101, 101 + n), dt_ms=20,
    )
    c = Circuit(params, ['Pyr', 'Pyr', 'PV', 'SST', 'VIP'], seed=27,
                readout='all', recurrent_gain=.4, type_probabilities=np.ones((4, 4)),
                calibration=(np.full(n, 1.5), np.linspace(1., 2., n), {}))
    c.W_out[:] = [.6, -.4, .3, -.2, .5]
    c.b_out = -.1
    return c


def batch_fixture(batch_size=5, time=8, seed=120):
    rng = np.random.default_rng(seed)
    x = rng.normal(0., .4, (batch_size, time, 5))
    x[:, time // 2:] = 0.
    y = np.where(np.arange(batch_size) % 2, 1., -1.)
    mask = np.zeros((batch_size, time))
    mask[:, -3:] = 1.
    if batch_size > 1:
        mask[0, -3] = 0.  # Different valid decision durations must not reweight trials.
    return SimpleNamespace(X=x, y=y, decision_mask=mask)


def objective(circuit, batch, noise=None):
    logits = circuit.forward(batch.X, current_noise=noise).logits
    weights = batch.decision_mask / batch.decision_mask.sum(1, keepdims=True)
    return float(np.mean(np.sum(np.logaddexp(0., -batch.y[:, None] * logits) * weights, axis=1)))


class LearningMathematicsTests(unittest.TestCase):
    def test_forward_kernel_matches_public_circuit_with_noise(self):
        c, batch = toy_circuit(), batch_fixture()
        noise = np.random.default_rng(8).normal(0, .15, (5, 8, 5))
        norms, _, logits = _forward_cache(c, batch.X, noise)
        trajectory = c.forward(batch.X, current_noise=noise)
        np.testing.assert_allclose(norms[:, 1:], trajectory.rates_norm, atol=1e-14, rtol=1e-14)
        np.testing.assert_allclose(logits, trajectory.logits, atol=1e-14, rtol=1e-14)

    def test_exact_bptt_matches_all_supported_finite_differences(self):
        c, batch = toy_circuit(), batch_fixture()
        noise = np.random.default_rng(82).normal(0, .1, (5, 8, 5))
        report = finite_difference_audit(c, batch, epsilon=1e-5, n_edges=100, noise=noise,
                                         rtol=2e-5, atol=2e-9)
        self.assertEqual(report['status'], 'pass', report)
        self.assertEqual(report['checked_recurrent_edges'], int(c.mask.sum()))
        self.assertLess(report['relative_l2_error'], 1e-7)
        for i in range(c.N):
            self.assertTrue(any(row.get('post') == i for row in report['checks']))
            self.assertTrue(any(row.get('pre') == i for row in report['checks']))

    def test_local_eligibility_matches_clamped_cell_finite_difference(self):
        """Numerically differentiate an independently defined clamped-cell loss.

        This tests the local Jacobian, NOT a false equivalence with the full RNN
        gradient.  All presynaptic trajectories are clamped, while the target
        cell's intrinsic beta feedback remains live.
        """
        c, batch = toy_circuit(), batch_fixture()
        normalized, _, logits = _forward_cache(c, batch.X)
        weights = batch.decision_mask / batch.decision_mask.sum(1, keepdims=True)
        teaching = (-batch.y[:, None] * np.exp(-np.logaddexp(0., batch.y[:, None] * logits))
                    * weights / len(batch.y))
        estimated = local_eligibility_gradients(c, batch, feedback_mode='random')['rec']

        def clamped_surrogate(i, j, magnitude):
            h = np.zeros((len(batch.y), c.K))
            rate = np.zeros(len(batch.y))
            row = c.W_rec[i].copy()
            row[j] = magnitude * c.signs[j]
            value = 0.
            for t in range(batch.X.shape[1]):
                drive = batch.X[:, t] @ c.input_weights[i] + normalized[:, t] @ row
                current = c.current_offset[i] + c.current_scale[i] * drive
                h = (1. - c.ds[i]) * h + c.a[i] * current[:, None] + c.beta[i] * rate[:, None]
                # Independent scalar polynomial, deliberately not calling the
                # learner's vectorized slope or eligibility implementation.
                z = (h.mean(1) - c.params.threshold[i]) / c.params.max_current[i]
                polynomial = sum(v ** 2 * z ** degree for degree, v in enumerate(c.params.poly_coeff[i]))
                rate = c.gamma_hz[i] * np.maximum(0., np.tanh(polynomial))
                value += np.sum(teaching[:, t] * c.feedback[i] * rate / c.rate_scale_hz[i])
            return value

        numeric = np.zeros_like(c.W_abs)
        for i, j in zip(c.edge_post, c.edge_pre):
            w = c.W_abs[i, j]
            eps = 1e-5
            numeric[i, j] = (clamped_surrogate(i, j, w + eps) - clamped_surrogate(i, j, w - eps)) / (2 * eps)
        np.testing.assert_allclose(estimated, numeric, atol=1e-10, rtol=2e-5)
        full = exact_bptt_gradients(c, batch)['rec']
        symmetric = local_eligibility_gradients(c, batch, feedback_mode='symmetric')['rec']
        self.assertGreater(np.linalg.norm(full - symmetric), 1e-7)

    def test_no_beta_ablation_retains_all_channels(self):
        c, batch = toy_circuit(beta=False), batch_fixture()
        full = local_eligibility_gradients(c, batch, include_beta=True)
        ablated = local_eligibility_gradients(c, batch, include_beta=False)
        np.testing.assert_array_equal(full['rec'], ablated['rec'])
        self.assertEqual(ablated['diagnostics']['eligibility_channels'], c.K)
        c = toy_circuit(beta=True)
        full = local_eligibility_gradients(c, batch, include_beta=True)
        ablated = local_eligibility_gradients(c, batch, include_beta=False)
        self.assertGreater(np.linalg.norm(full['rec'] - ablated['rec']), 1e-6)
        self.assertEqual(full['loss'], ablated['loss'])
        np.testing.assert_array_equal(full['out'], ablated['out'])

    def test_all_deterministic_rules_share_loss_and_readout_derivative(self):
        c, batch = toy_circuit(), batch_fixture()
        noise = np.random.default_rng(16).normal(0, .1, (5, 8, 5))
        exact = exact_bptt_gradients(c, batch, noise=noise)
        for other in (local_eligibility_gradients(c, batch, noise=noise),
                      readout_only_gradients(c, batch, noise=noise)):
            self.assertAlmostEqual(exact['loss'], other['loss'], places=14)
            self.assertEqual(exact['accuracy'], other['accuracy'])
            np.testing.assert_allclose(exact['out'], other['out'], rtol=1e-14, atol=1e-14)
            self.assertAlmostEqual(exact['bias'], other['bias'], places=14)
        self.assertTrue(np.any(exact['rec'][c.signs < 0] != 0))
        self.assertTrue(np.any(local_eligibility_gradients(c, batch)['rec'][c.signs < 0] != 0))

    def test_estimators_do_not_mutate_circuit_or_physiology(self):
        c, batch = toy_circuit(), batch_fixture()
        before = c.state_dict()
        fingerprint = c.params.fingerprint()
        exact_bptt_gradients(c, batch)
        local_eligibility_gradients(c, batch)
        node_perturbation_gradients(c, batch, rng=np.random.default_rng(13))
        readout_only_gradients(c, batch)
        for key, value in before.items():
            np.testing.assert_array_equal(c.state_dict()[key], value)
        self.assertEqual(fingerprint, c.params.fingerprint())
        c.assert_constraints()

    def test_small_exact_gradient_step_decreases_same_objective(self):
        c, batch = toy_circuit(), batch_fixture()
        before = objective(c, batch)
        grad = exact_bptt_gradients(c, batch)
        opt = ProjectedOptimizer(kind='sgd', lr_rec=.01, lr_out=.01, max_grad_norm=None)
        opt.step(c, grad)
        self.assertLess(objective(c, batch), before)

    def test_node_perturbation_score_matches_noisy_path_gradient_in_expectation(self):
        """Monte Carlo score/pathwise identity, including sigma^2 and Dale signs.

        Independent replicates provide an empirical standard error.  A generous
        six-SE bound avoids pretending finite Monte Carlo is exact.  This is a
        correctness diagnostic on a tiny model, not a benchmark performance run.
        """
        c = toy_circuit()
        base = batch_fixture(batch_size=1, time=5)
        n = 384
        batch = SimpleNamespace(X=np.repeat(base.X, n, axis=0),
                                y=np.repeat(base.y, n),
                                decision_mask=np.repeat(base.decision_mask, n, axis=0))
        sigma, differences, expected = .18, [], []
        for seed in range(32):
            score = node_perturbation_gradients(c, batch, rng=np.random.default_rng(seed), sigma=sigma)
            # The online estimator draws a [B,N] block at each time. Reproduce
            # exactly that noise sequence for the conditional BPTT derivative.
            draw = np.random.default_rng(seed)
            noise = np.stack([draw.normal(0, sigma, (n, c.N)) for _ in range(5)], axis=1)
            path = exact_bptt_gradients(c, batch, noise=noise)
            differences.append(score['rec'][c.mask] - path['rec'][c.mask])
            expected.append(path['rec'][c.mask])
        differences = np.array(differences)
        se = differences.std(0, ddof=1) / np.sqrt(len(differences))
        self.assertTrue(np.all(np.abs(differences.mean(0)) < 6 * se + 1e-7))
        self.assertGreater(np.linalg.norm(np.mean(expected, axis=0)), 1e-4)
        score_mean = np.mean(expected, axis=0) + differences.mean(0)
        truth = np.mean(expected, axis=0)
        cosine = np.dot(score_mean, truth) / (np.linalg.norm(score_mean) * np.linalg.norm(truth))
        self.assertGreater(cosine, .98)

    def test_node_reproducibility_and_cost_count(self):
        c, batch = toy_circuit(), batch_fixture()
        a = node_perturbation_gradients(c, batch, rng=np.random.default_rng(42))
        b = node_perturbation_gradients(c, batch, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(a['rec'], b['rec'])
        self.assertEqual(a['diagnostics']['forward_rollouts'], 2)
        self.assertEqual(a['diagnostics']['simulated_trial_bins'], 2 * batch.X.shape[0] * batch.X.shape[1])
        with self.assertRaises(ValueError):
            node_perturbation_gradients(c, batch, rng=np.random.default_rng(0), sigma=0)

    def test_padding_after_response_adds_no_node_score_noise(self):
        c, batch = toy_circuit(), batch_fixture()
        padded = SimpleNamespace(X=np.pad(batch.X, ((0, 0), (0, 4), (0, 0))),
                                 y=batch.y.copy(),
                                 decision_mask=np.pad(batch.decision_mask, ((0, 0), (0, 4))))
        before = node_perturbation_gradients(c, batch, rng=np.random.default_rng(2))
        after = node_perturbation_gradients(c, padded, rng=np.random.default_rng(2))
        np.testing.assert_array_equal(before['rec'], after['rec'])
        np.testing.assert_array_equal(before['out'], after['out'])
        self.assertEqual(before['loss'], after['loss'])

    def test_invalid_decisions_and_labels_fail(self):
        c, batch = toy_circuit(), batch_fixture()
        batch.y[0] = 0.
        with self.assertRaises(ValueError):
            exact_bptt_gradients(c, batch)
        batch.y[0] = -1.
        batch.decision_mask[0] = 0.
        with self.assertRaises(ValueError):
            local_eligibility_gradients(c, batch)

    def test_flat_recurrent_check_is_inconclusive(self):
        c, batch = toy_circuit(), batch_fixture()
        c.W_out[:] = 0.
        report = finite_difference_audit(c, batch, n_edges=20)
        self.assertEqual(report['status'], 'inconclusive')
        self.assertEqual(report['nonzero_recurrent_checks'], 0)

    def test_optimizer_constraints_frozen_parameters_and_resume(self):
        c, batch = toy_circuit(), batch_fixture()
        fingerprint = c.params.fingerprint()
        opt = ProjectedOptimizer(kind='adam', lr_rec=.03, lr_out=.04, max_weight=.15)
        opt.step(c, exact_bptt_gradients(c, batch))
        self.assertTrue(np.all(c.W_abs >= 0))
        self.assertTrue(np.all(c.W_abs <= .15))
        c.assert_constraints()
        self.assertEqual(c.params.fingerprint(), fingerprint)
        c2 = copy.deepcopy(c)
        opt2 = ProjectedOptimizer()
        opt2.load_state_dict(opt.state_dict())
        result1 = opt.step(c, exact_bptt_gradients(c, batch))
        result2 = opt2.step(c2, exact_bptt_gradients(c2, batch))
        np.testing.assert_array_equal(c.W_abs, c2.W_abs)
        np.testing.assert_array_equal(c.W_out, c2.W_out)
        self.assertEqual(c.b_out, c2.b_out)
        self.assertEqual(result1, result2)

    def test_reservoir_recurrence_stays_frozen_even_with_weight_decay(self):
        c, batch = toy_circuit(), batch_fixture()
        initial = c.W_abs.copy()
        opt = ProjectedOptimizer(lr_rec=.1, lr_out=.1, weight_decay=.5, max_weight=.01)
        opt.step(c, readout_only_gradients(c, batch))
        np.testing.assert_array_equal(initial, c.W_abs)


class AuthenticCellIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = load_dataset(ROOT / 'model/gfr_dataset.json', ROOT / 'data/metadata.csv')
        params, rows = sample_cells(df, counts=(4, 2, 2, 2), seed=17)
        cls.circuit = Circuit(params, rows.cell_type, seed=29, readout='all',
                              type_probabilities=np.ones((4, 4)), input_gain=.7)
        cls.batch = batch_fixture(batch_size=4, time=14, seed=70)

    def test_authentic_eight_channel_forward_and_bptt(self):
        c, batch = self.circuit, self.batch
        self.assertEqual(c.K, 8)
        normalized, _, logits = _forward_cache(c, batch.X)
        root = c.forward(batch.X)
        np.testing.assert_allclose(root.rates_norm, normalized[:, 1:], rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(root.logits, logits, rtol=1e-14, atol=1e-14)
        report = finite_difference_audit(c, batch, n_edges=30, epsilon=1e-5,
                                         rtol=5e-4, atol=1e-8)
        self.assertEqual(report['status'], 'pass', report)
        self.assertLess(report['relative_l2_error'], 1e-5)
        posts = [row['post'] for row in report['checks'] if row['parameter'] == 'rec']
        sources = [row['pre'] for row in report['checks'] if row['parameter'] == 'rec']
        self.assertTrue(np.any(c.signs[posts] < 0))
        self.assertTrue(np.any(c.signs[posts] > 0))
        self.assertTrue(np.any(c.signs[sources] < 0))
        self.assertTrue(np.any(c.signs[sources] > 0))


if __name__ == '__main__':
    unittest.main()
