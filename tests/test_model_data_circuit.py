"""Scientific invariants of the fitted cell bank, circuit controls and state map.

The upstream oracle below is an independent scalar translation of the pinned
Allen PyTorch equation, evaluated in float64 with rates in spikes/ms. It does
not import the implementation under test. This is equation compatibility,
not a claim that PyTorch was installed or raw electrophysiology was rerun.
"""
from __future__ import annotations

import copy
from dataclasses import fields, replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal

from grnn_bio.model import (GFRParameters, activation_and_grad, cell_step,
                            isolated_response, calibrate_current)
from grnn_bio.data import (CELL_TYPES, load_dataset, parameters_from_rows,
                           partition_cells, sample_cells, manipulate_identity)
from grnn_bio.circuit import Circuit, degree_preserving_rewire
from grnn_bio.dynamics import (constant_input_step, fixed_point_residual,
                               full_state_jacobian, fixed_point_diagnostics,
                               find_fixed_point)
from grnn_bio.randomness import rng_for, seed_for


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / 'model' / 'gfr_dataset.json'
METADATA = ROOT / 'data' / 'metadata.csv'


def independent_upstream_rates(raw_records, currents_pa):
    """Literal cell/filter loops, direct polynomial, original rate units.

    Reference: AllenInstitute/GRNN model.py at
    835a535f6831b88339c6a5c17180fdbc504f7448, GFR.forward and
    PolynomialActivation.forward. Deliberately avoids our loader and Horner
    implementation, and retains the upstream 1000 conversion inside history.
    """
    batch, steps, count = currents_pa.shape
    output = np.zeros_like(currents_pa)
    for i, entry in enumerate(raw_records):
        p, g = entry['params'], entry['params']['g']
        k = len(p['ds'])
        state = np.zeros((batch, k))
        previous_spikes_per_ms = np.zeros(batch)
        for t in range(steps):
            for filt in range(k):
                state[:, filt] = ((1 - p['ds'][filt]) * state[:, filt]
                                  + p['a'][0][filt] * currents_pa[:, t, i]
                                  + 1000. * p['b'][0][filt] * previous_spikes_per_ms)
            normalized = (state.sum(axis=1) / k - g['b'][0]) / g['max_current'][0]
            poly = np.zeros(batch)
            for power, coefficient in enumerate(g['poly_coeff'][0]):
                poly += coefficient ** 2 * normalized ** power
            previous_spikes_per_ms = g['max_firing_rate'][0] * np.maximum(np.tanh(poly), 0.)
            output[:, t, i] = 1000. * previous_spikes_per_ms
    assert count == len(raw_records)
    return output


def toy_params(n=4, degree=1, k=2):
    coeff = np.zeros((n, degree + 1))
    coeff[:, 0] = .2
    if degree >= 1:
        coeff[:, 1] = 1.
    return GFRParameters(
        a=np.tile(np.linspace(.7, .2, k), (n, 1)),
        beta=np.tile(np.linspace(-.01, -.03, k), (n, 1)),
        ds=np.tile(np.linspace(1., .25, k), (n, 1)),
        max_current=np.full(n, 100.), gamma_hz=np.linspace(30., 60., n),
        poly_coeff=coeff, threshold=np.linspace(0., 5., n),
        cell_ids=np.arange(100, 100+n), dt_ms=20)


def toy_circuit(params=None, **kwargs):
    params = toy_params() if params is None else params
    return Circuit(params, list(CELL_TYPES), seed=23,
                   calibration=(np.full(params.n, 30.), np.full(params.n, 20.),
                                {'uses_task_labels': False, 'test_fixture': True}),
                   type_probabilities=np.full((4, 4), .8), **kwargs)


class UpstreamEquationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = json.loads(DATASET.read_text())
        cls.bank = parameters_from_rows(pd.DataFrame(cls.records))

    def test_all_1124_imported_tuples_are_unchanged(self):
        self.assertEqual(len(self.records), 1124)
        self.assertEqual(self.bank.k, 8)
        self.assertEqual(self.bank.dt_ms, 20)
        for i, entry in enumerate(self.records):
            p, g = entry['params'], entry['params']['g']
            with self.subTest(cell=entry['cell_id']):
                assert_array_equal(self.bank.a[i], p['a'][0])
                assert_array_equal(self.bank.beta[i], p['b'][0])
                assert_array_equal(self.bank.ds[i], p['ds'])
                assert_array_equal(self.bank.poly_coeff[i], g['poly_coeff'][0])
                self.assertEqual(self.bank.max_current[i], g['max_current'][0])
                self.assertEqual(self.bank.threshold[i], g['b'][0])
                self.assertEqual(self.bank.gamma_hz[i], 1000. * g['max_firing_rate'][0])
                self.assertEqual(self.bank.cell_ids[i], entry['cell_id'])
        self.assertTrue(np.any(self.bank.a[:, 1:] < 0), 'Signed fitted slow taps must survive')
        self.assertTrue(np.any(self.bank.beta < 0), 'Fitted history terms must survive')

    def test_full_bank_matches_independent_spikes_per_ms_reference(self):
        rng = np.random.default_rng(1843)
        currents = rng.uniform(-.1, 1.1, (2, 11, self.bank.n)) * self.bank.max_current
        expected = independent_upstream_rates(self.records, currents)
        observed = isolated_response(self.bank, currents)
        assert_allclose(observed, expected, rtol=2e-11, atol=2e-10)

    def test_no_second_timestep_scaling(self):
        p = toy_params()
        h = np.arange(p.n*p.k, dtype=float).reshape(p.n, p.k) + 2.
        r = np.linspace(3., 9., p.n)
        current = np.linspace(11., 23., p.n)
        expected_h = (1. - p.ds)*h + p.a*current[:, None] + p.beta*r[:, None]
        got, _, _ = cell_step(p, h, r, current)
        assert_allclose(got, expected_h, rtol=0., atol=1e-14)
        # Coefficients represent the already-discretized step. Merely relabeling
        # metadata cannot insert a timestep into this same operator.
        alternative_dt, _, _ = cell_step(replace(p, dt_ms=100), h, r, current)
        assert_array_equal(got, alternative_dt)

    def test_initial_response_uses_zero_state_and_zero_history(self):
        p = toy_params()
        current = np.full((3, 1, p.n), 30.)
        expected, _ = activation_and_grad((p.a*30.).mean(axis=1), p)
        assert_allclose(isolated_response(p, current)[:, 0], np.tile(expected, (3, 1)))

    def test_fitted_decay_basis(self):
        expected = np.r_[1., 1. - np.exp(-20./np.array([20, 50, 100, 200, 500, 1000, 2000]))]
        # Upstream stores the float32 fitted basis; do not demand more precision.
        assert_allclose(self.bank.ds, np.tile(expected, (self.bank.n, 1)), atol=6e-8, rtol=0.)


class ActivationAndCalibrationTests(unittest.TestCase):
    def test_constant_activation_has_zero_derivative(self):
        p = toy_params(degree=0)
        z = np.linspace(-900., 900., 28).reshape(7, 4)
        rate, grad = activation_and_grad(z, p)
        expected = p.gamma_hz*np.tanh(p.poly_coeff[:, 0]**2)
        assert_allclose(rate, np.tile(expected, (7, 1)))
        assert_array_equal(grad, np.zeros_like(z))

    def test_higher_degree_polynomial_and_derivative(self):
        coeff = np.array([[.1, -.8, .25, .03], [-.2, 1.2, -.15, .04]])
        p = replace(toy_params(n=2), max_current=np.array([2., 3.]),
                    threshold=np.array([.2, -.4]), poly_coeff=coeff)
        z = np.array([[-3., -4.], [.1, .2], [2., 3.], [5., 6.]])
        x = (z-p.threshold)/p.max_current
        poly = sum(coeff[:, d]**2*x**d for d in range(4))
        expected = p.gamma_hz*np.maximum(np.tanh(poly), 0.)
        rate, grad = activation_and_grad(z, p)
        assert_allclose(rate, expected, rtol=2e-14, atol=1e-14)
        eps = 1e-5
        numerical = (activation_and_grad(z+eps, p)[0]-activation_and_grad(z-eps, p)[0])/(2*eps)
        assert_allclose(grad, numerical, rtol=2e-7, atol=2e-8)

    def test_relu_inactive_branch_and_exact_kink(self):
        p = replace(toy_params(), poly_coeff=np.tile([0., 1.], (4, 1)),
                    threshold=np.zeros(4))
        rate, derivative = activation_and_grad(np.full((2, 4), -10.), p)
        assert_array_equal(rate, np.zeros((2, 4)))
        assert_array_equal(derivative, np.zeros((2, 4)))
        _, kink_derivative = activation_and_grad(np.zeros(4), p)
        assert_array_equal(kink_derivative, np.zeros(4))

    def test_immutable_parameters_own_their_data(self):
        p = toy_params()
        a = p.a.copy()
        cloned = replace(p, a=a)
        fingerprint = cloned.fingerprint()
        a[0, 0] += 90.
        self.assertEqual(cloned.fingerprint(), fingerprint)
        for f in fields(cloned):
            if f.name != 'dt_ms':
                with self.subTest(parameter=f.name), self.assertRaises(ValueError):
                    getattr(cloned, f.name).flat[0] = 1.

    def test_calibration_is_label_free_and_preserves_parameters(self):
        p = toy_params()
        fingerprint = p.fingerprint()
        offset, scale, info = calibrate_current(p, baseline_fraction=.1,
                                                input_scale_fraction=.3, n_grid=9,
                                                probe_steps=18)
        self.assertEqual(p.fingerprint(), fingerprint)
        self.assertFalse(info['uses_task_labels'])
        assert_allclose(scale, .3*p.max_current)
        self.assertTrue(np.all((offset >= 0.) & (offset <= p.max_current)))
        grid = np.linspace(0., 1., 9)[:, None]*p.max_current
        probes = np.broadcast_to(grid[:, None], (9, 18, p.n))
        steady = isolated_response(p, probes)[:, 12:].mean(axis=1)
        chosen = np.argmin(abs(steady-.1*p.gamma_hz), axis=0)
        assert_array_equal(offset, grid[chosen, np.arange(p.n)])
        assert_allclose(info['achieved_rate_hz'], steady[chosen, np.arange(p.n)])

    def test_nonphysical_parameters_are_rejected(self):
        p = toy_params()
        for update in ({'gamma_hz': np.zeros(4)}, {'max_current': np.full(4, -1.)},
                       {'ds': np.zeros((4, 2))}, {'a': np.full((4, 2), np.nan)}):
            with self.subTest(update=list(update)), self.assertRaises(ValueError):
                replace(p, **update)

    def test_empty_cell_filter_or_polynomial_axes_are_rejected(self):
        p = toy_params()
        invalid = [
            {'a': np.zeros((4, 0)), 'beta': np.zeros((4, 0)), 'ds': np.zeros((4, 0))},
            {'a': np.zeros((0, 2)), 'beta': np.zeros((0, 2)), 'ds': np.zeros((0, 2))},
            {'a': np.zeros(4)},
            {'poly_coeff': np.zeros((4, 0))},
        ]
        for update in invalid:
            with self.subTest(shapes={k: v.shape for k, v in update.items()}), self.assertRaises(ValueError):
                replace(p, **update)

    def test_timestep_requires_positive_integer_type(self):
        p = toy_params()
        for dt in [0, -1, 20.5, 20., np.nan, np.inf, -np.inf, True, '20']:
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                replace(p, dt_ms=dt)
        self.assertEqual(replace(p, dt_ms=np.int64(20)).dt_ms, 20)

    def test_calibration_rejects_nonfinite_fractions_and_noninteger_grids(self):
        p = toy_params()
        for kwargs in [{'baseline_fraction': np.nan}, {'input_scale_fraction': np.inf},
                       {'input_scale_fraction': np.nan}, {'n_grid': 4.5},
                       {'probe_steps': np.inf}, {'probe_steps': True}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                calibrate_current(p, **kwargs)


class SelectionAndIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_dataset(DATASET, METADATA)
        cls.params, cls.rows = sample_cells(cls.data, counts=(12, 4, 3, 3), seed=294)

    def test_cell_admission_does_not_use_test_evr(self):
        raw = json.loads(DATASET.read_text())
        for idx, row in enumerate(raw):
            row['test_evr'] = -1000. if idx % 2 else 1000.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'changed_test_scores.json'
            path.write_text(json.dumps(raw))
            changed = load_dataset(path, METADATA)
        assert_array_equal(changed.cell_id, self.data.cell_id)
        assert_array_equal(changed.cell_type, self.data.cell_type)

    def test_specimen_cohorts_are_stratified_disjoint_and_exhaustive(self):
        development = partition_cells(self.data, 'development', seed=71)
        confirmation = partition_cells(self.data, 'confirmation', seed=71)
        left, right = set(development.cell_id), set(confirmation.cell_id)
        self.assertFalse(left & right)
        self.assertEqual(left | right, set(self.data.cell_id))
        self.assertEqual(set(development.cell_type), set(CELL_TYPES))
        self.assertEqual(set(confirmation.cell_type), set(CELL_TYPES))
        assert_array_equal(confirmation.cell_id, partition_cells(self.data, 'confirmation', seed=71).cell_id)

    def test_sampling_has_unique_cells_and_explicit_insufficient_pool_error(self):
        self.assertEqual(len(set(self.rows.cell_id)), len(self.rows))
        counts = self.rows.groupby('cell_type').size().to_dict()
        self.assertEqual(counts, {'Pyr': 12, 'PV': 4, 'SST': 3, 'VIP': 3})
        impossible = (len(self.data[self.data.cell_type == 'Pyr'])+1, 1, 1, 1)
        with self.assertRaises(ValueError):
            sample_cells(self.data, counts=impossible)
        _, same = sample_cells(self.data, counts=(12, 4, 3, 3), seed=294)
        assert_array_equal(same.cell_id, self.rows.cell_id)

    def test_wrong_discretization_and_duplicate_ids_fail(self):
        with self.assertRaises(ValueError):
            load_dataset(DATASET, METADATA, dt_ms=10)
        raw = json.loads(DATASET.read_text())
        raw.append(copy.deepcopy(raw[0]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'duplicated_cells.json'
            path.write_text(json.dumps(raw))
            with self.assertRaises(ValueError):
                load_dataset(path, METADATA)

    def test_authentic_bank_is_exact_and_no_control_mutates_source(self):
        p = self.params
        fingerprint = p.fingerprint()
        for mode in ['authentic', 'within_type_shuffle', 'cross_type_permutation',
                     'type_homogeneous', 'fast_decay']:
            with self.subTest(mode=mode):
                out, metadata = manipulate_identity(p, self.rows.cell_type, mode, seed=81)
                self.assertEqual(p.fingerprint(), fingerprint)
                self.assertEqual(metadata['reference_fingerprint'], fingerprint)
                if mode == 'authentic':
                    self.assertEqual(out.fingerprint(), fingerprint)
                    self.assertFalse(metadata['synthetic'])

    def test_cross_type_permutation_preserves_intact_tuples_bijectively(self):
        p, types = self.params, self.rows.cell_type
        out, metadata = manipulate_identity(p, types, 'cross_type_permutation', seed=12)
        self.assertEqual(sorted(out.cell_ids), sorted(p.cell_ids))
        lookup = {int(cell): i for i, cell in enumerate(p.cell_ids)}
        for target, cell_id in enumerate(out.cell_ids):
            source = lookup[int(cell_id)]
            for f in fields(p):
                if f.name != 'dt_ms':
                    assert_array_equal(getattr(out, f.name)[target], getattr(p, f.name)[source])
            self.assertEqual(metadata['source_cell_ids']['tuple'][target], int(cell_id))

    def test_independent_shuffle_preserves_within_class_marginals_and_activation_tuples(self):
        p = self.params
        out, metadata = manipulate_identity(p, self.rows.cell_type, 'within_type_shuffle', seed=12)
        lookup = {int(cell): i for i, cell in enumerate(p.cell_ids)}
        groups = {'a': ['a'], 'beta': ['beta'],
                  'activation': ['max_current', 'gamma_hz', 'poly_coeff', 'threshold']}
        for label in CELL_TYPES:
            indices = np.flatnonzero(self.rows.cell_type.to_numpy() == label)
            for group, names in groups.items():
                source_ids = np.asarray(metadata['source_cell_ids'][group])[indices]
                self.assertEqual(sorted(source_ids), sorted(p.cell_ids[indices]))
                for target, source_id in zip(indices, source_ids):
                    source = lookup[int(source_id)]
                    for name in names:
                        assert_array_equal(getattr(out, name)[target], getattr(p, name)[source])

    def test_homogeneous_condition_is_exact_documented_within_class_mean(self):
        p = self.params
        out, metadata = manipulate_identity(p, self.rows.cell_type, 'type_homogeneous')
        self.assertTrue(metadata['synthetic'])
        for label in CELL_TYPES:
            idx = np.flatnonzero(self.rows.cell_type.to_numpy() == label)
            for name in ['a', 'beta', 'ds', 'max_current', 'gamma_hz', 'poly_coeff', 'threshold']:
                mean = getattr(p, name)[idx].mean(axis=0)
                assert_allclose(getattr(out, name)[idx], np.broadcast_to(mean, getattr(out, name)[idx].shape))
        assert_array_equal(metadata['source_cell_ids']['tuple'], np.full(p.n, -1))

    def test_fast_decay_preserves_both_static_filter_ratios(self):
        p = self.params
        out, _ = manipulate_identity(p, self.rows.cell_type, 'fast_decay')
        assert_allclose(out.a/out.ds, p.a/p.ds, rtol=3e-16, atol=2e-15)
        assert_allclose(out.beta/out.ds, p.beta/p.ds, rtol=3e-16, atol=2e-15)
        self.assertTrue(np.all(out.ds >= p.ds))
        assert_array_equal(out.ds[:, 0], p.ds[:, 0])
        for name in ['max_current', 'gamma_hz', 'poly_coeff', 'threshold']:
            assert_array_equal(getattr(out, name), getattr(p, name))


class CircuitControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = load_dataset(DATASET, METADATA)
        cls.params, cls.rows = sample_cells(data, counts=(9, 3, 2, 2), seed=31)
        cls.calibration = calibrate_current(cls.params, n_grid=7, probe_steps=15)

    def make_circuit(self, **kwargs):
        return Circuit(self.params, self.rows.cell_type, seed=832,
                       calibration=self.calibration, **kwargs)

    def test_non100_network_has_no_hidden_fixed_population_slices(self):
        c = self.make_circuit(input_mode='vip_context')
        self.assertEqual(c.N, 16)
        self.assertEqual(c.W_abs.shape, (16, 16))
        not_vip = c.cell_types != 'VIP'
        assert_array_equal(c.input_weights[not_vip, 2:4], np.zeros((not_vip.sum(), 2)))
        self.assertTrue(np.any(c.input_weights[~not_vip, 2:4] != 0))
        trajectory = c.forward(np.zeros((3, 7, 5)), record_states=True, record_currents=True)
        self.assertEqual(trajectory.states.shape, (3, 7, 16, 8))
        self.assertTrue(np.isfinite(trajectory.rates_hz).all())
        self.assertTrue(np.all(trajectory.rates_hz >= 0))
        self.assertTrue(np.all(trajectory.rates_hz <= c.gamma_hz))
        c.assert_constraints()

    def test_zero_initial_state_and_first_current_no_recurrent_prewarm(self):
        c = self.make_circuit()
        h, rates = c.initial_state(2)
        assert_array_equal(h, np.zeros((2, c.N, c.K)))
        assert_array_equal(rates, np.zeros((2, c.N)))
        X = np.random.default_rng(2).normal(size=(2, 1, 5))
        result = c.forward(X, record_currents=True, record_states=True)
        expected_current = c.current_offset + c.current_scale*(X[:, 0] @ c.input_weights.T)
        assert_allclose(result.currents_pa[:, 0], expected_current, atol=1e-12)
        assert_allclose(result.states[:, 0], c.a*expected_current[:, :, None], atol=1e-12)

    def test_directed_rewiring_preserves_both_degrees_without_touching_source(self):
        rng = np.random.default_rng(16)
        m = rng.random((25, 25)) < .25
        np.fill_diagonal(m, False)
        before = m.copy()
        for seed in range(3):
            with self.subTest(seed=seed):
                rewritten, info = degree_preserving_rewire(m, np.random.default_rng(seed), swaps_per_edge=3)
                assert_array_equal(rewritten.sum(0), m.sum(0))
                assert_array_equal(rewritten.sum(1), m.sum(1))
                self.assertFalse(np.diag(rewritten).any())
                self.assertGreater(info['accepted_swaps'], 0)
                self.assertFalse(info['uniform_null_sampling_claimed'])
        assert_array_equal(m, before)

    def test_topology_control_preserves_per_column_magnitude_multiset_and_sign(self):
        reference = self.make_circuit()
        rewired = self.make_circuit(topology='degree_rewired', swaps_per_edge=3)
        assert_array_equal(reference.mask.sum(0), rewired.mask.sum(0))
        assert_array_equal(reference.mask.sum(1), rewired.mask.sum(1))
        for j in range(reference.N):
            assert_array_equal(np.sort(reference.W_abs[reference.mask[:, j], j]),
                               np.sort(rewired.W_abs[rewired.mask[:, j], j]))
        assert_allclose(reference.W_abs.sum(0), rewired.W_abs.sum(0), atol=1e-14)
        self.assertTrue(np.all(rewired.W_rec[:, rewired.signs > 0] >= 0))
        self.assertTrue(np.all(rewired.W_rec[:, rewired.signs < 0] <= 0))
        random = self.make_circuit(topology='edge_count_random')
        self.assertEqual(reference.mask.sum(), random.mask.sum())

    def test_all_controls_share_input_readout_feedback_currents_and_physical_rate_scale(self):
        reference = self.make_circuit()
        for identity in ['authentic', 'within_type_shuffle', 'cross_type_permutation',
                         'type_homogeneous', 'fast_decay']:
            for topology in ['type_prior', 'degree_rewired', 'edge_count_random']:
                with self.subTest(identity=identity, topology=topology):
                    c = self.make_circuit(identity=identity, topology=topology, swaps_per_edge=1)
                    for name in ['input_weights', 'W_out', 'feedback', 'current_offset',
                                 'current_scale', 'rate_scale_hz', 'signs', 'readout_mask']:
                        assert_array_equal(getattr(c, name), getattr(reference, name))
                    assert_array_equal(c.rate_scale_hz, np.full(c.N, 100.))

    def test_constraints_detect_feedback_mask_sign_and_readout_mutations(self):
        c = self.make_circuit()
        c.feedback[0] += .1
        with self.assertRaises(AssertionError):
            c.assert_constraints()
        c = self.make_circuit()
        c.W_abs[0, 0] = .1
        with self.assertRaises(AssertionError):
            c.assert_constraints()
        c = self.make_circuit()
        i, j = np.argwhere(c.mask)[0]
        c.W_abs[i, j] = -.1
        with self.assertRaises(AssertionError):
            c.assert_constraints()
        c = self.make_circuit()
        c.W_out[np.flatnonzero(~c.readout_mask)[0]] = .1
        with self.assertRaises(AssertionError):
            c.assert_constraints()

    def test_checkpoint_roundtrip_and_construction_mismatch_rejection(self):
        c = self.make_circuit()
        c.W_abs *= .8
        c.W_out *= 1.2
        c.b_out = .7
        state = c.state_dict()
        restored = self.make_circuit()
        restored.load_state_dict(state)
        X = np.random.default_rng(3).normal(size=(2, 5, 5))
        assert_array_equal(c.forward(X).logits, restored.forward(X).logits)
        for mismatch in ['intrinsic_fingerprint', 'mask', 'input_weights', 'feedback',
                         'current_offset', 'current_scale', 'rate_scale_hz']:
            bad = copy.deepcopy(state)
            if mismatch == 'intrinsic_fingerprint':
                bad[mismatch] = np.array('wrong-fitted-cell-bank')
            elif mismatch == 'mask':
                bad[mismatch][0, 0] = ~bad[mismatch][0, 0]
            else:
                bad[mismatch].flat[0] += 1.
            with self.subTest(mismatch=mismatch), self.assertRaises(ValueError):
                restored.load_state_dict(bad)
        for name in ['W_abs', 'W_out']:
            bad = copy.deepcopy(state)
            bad[name] = bad[name][:-1]
            with self.subTest(shape=name), self.assertRaises(ValueError):
                restored.load_state_dict(bad)

    def test_checkpoint_rejects_changed_readout_support_slot_labels_and_signs(self):
        source = self.make_circuit()
        state = source.state_dict()
        changed_readout = self.make_circuit(readout='all')
        with self.assertRaisesRegex(ValueError, 'readout_mask'):
            changed_readout.load_state_dict(state)
        for name in ['signs', 'cell_types']:
            changed = copy.deepcopy(state)
            if name == 'signs':
                changed[name][0] *= -1.
            else:
                changed[name][0] = 'PV'
            with self.subTest(field=name), self.assertRaisesRegex(ValueError, name):
                source.load_state_dict(changed)
        incomplete = copy.deepcopy(state)
        del incomplete['readout_mask']
        with self.assertRaisesRegex(ValueError, 'missing required fields'):
            source.load_state_dict(incomplete)

    def test_rejected_checkpoint_does_not_partially_change_any_state(self):
        c = self.make_circuit()
        before = c.state_dict()
        for defect in ['late_shape', 'nonfinite_bias', 'bias_shape', 'negative_weight',
                       'masked_weight', 'readout_support', 'nonfinite_readout']:
            bad = copy.deepcopy(before)
            # This valid earlier field would expose the old partial-update bug.
            bad['W_abs'] *= .5
            if defect == 'late_shape':
                bad['W_out'] = bad['W_out'][:-1]
            elif defect == 'nonfinite_bias':
                bad['b_out'] = np.array(np.nan)
            elif defect == 'bias_shape':
                bad['b_out'] = np.array([0.])
            elif defect == 'negative_weight':
                i, j = np.argwhere(c.mask)[0]
                bad['W_abs'][i, j] = -.1
            elif defect == 'masked_weight':
                bad['W_abs'][0, 0] = .1
            elif defect == 'readout_support':
                bad['W_out'][np.flatnonzero(~c.readout_mask)[0]] = .1
            elif defect == 'nonfinite_readout':
                bad['W_out'][0] = np.inf
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                c.load_state_dict(bad)
            for key, value in before.items():
                assert_array_equal(c.state_dict()[key], value)

    def test_nonfinite_connection_gains_and_probability_priors_are_rejected(self):
        for name in ['recurrent_gain', 'inhibitory_gain', 'input_gain']:
            for value in [np.nan, np.inf, -np.inf]:
                with self.subTest(gain=name, value=value), self.assertRaises(ValueError):
                    self.make_circuit(**{name: value})
        for value in [np.nan, np.inf, -np.inf]:
            prior = np.full((4, 4), .5)
            prior[1, 2] = value
            with self.subTest(probability=value), self.assertRaises(ValueError):
                self.make_circuit(type_probabilities=prior)

    def test_rate_clamp_enters_subsequent_history_and_readout(self):
        c = self.make_circuit()
        x = np.zeros((2, 4, 5))
        trajectory = c.forward(x, rate_multiplier=np.zeros(c.N), record_currents=True)
        assert_array_equal(trajectory.rates_hz, np.zeros((2, 4, c.N)))
        assert_array_equal(trajectory.logits, np.full((2, 4), c.b_out))
        assert_allclose(trajectory.currents_pa,
                        np.broadcast_to(c.current_offset, (2, 4, c.N)), atol=1e-14)


class FullStateDynamicsTests(unittest.TestCase):
    def test_full_jacobian_matches_finite_differences_including_network_paths(self):
        c = toy_circuit()
        h = np.arange(c.N*c.K, dtype=float).reshape(c.N, c.K) + 20.
        x = np.array([.3, -.2, .1, .2, 0.])
        numerical = np.zeros((h.size, h.size))
        eps = 2e-5
        for j in range(h.size):
            perturb = np.zeros_like(h)
            perturb.flat[j] = eps
            numerical[:, j] = ((constant_input_step(c, h+perturb, x)
                                 - constant_input_step(c, h-perturb, x))/(2*eps)).ravel()
        analytical = full_state_jacobian(c, h, x)
        assert_allclose(analytical, numerical, rtol=2e-7, atol=2e-9)
        # At least one nonlocal block must be nonzero: this is the full circuit,
        # not just a collection of cell-local Jacobians.
        off_diagonal = analytical.copy()
        for i in range(c.N):
            sl = slice(i*c.K, (i+1)*c.K)
            off_diagonal[sl, sl] = 0.
        self.assertGreater(np.linalg.norm(off_diagonal), 0.)

    def test_known_stationary_state_and_spectral_radius(self):
        p = toy_params(degree=0)
        p = replace(p, a=np.zeros_like(p.a), beta=np.zeros_like(p.beta),
                    ds=np.full_like(p.ds, .4))
        c = toy_circuit(p)
        h = np.zeros((c.N, c.K))
        assert_array_equal(fixed_point_residual(c, h), np.zeros_like(h))
        report = fixed_point_diagnostics(c, h)
        self.assertTrue(report['is_approximate_fixed_point'])
        self.assertTrue(report['linearly_stable_fixed_point'])
        self.assertAlmostEqual(report['full_state_spectral_radius'], .6)
        self.assertEqual(report['state_dimension'], c.N*c.K)
        self.assertFalse(report['global_attractor_claimed'])

    def test_transient_or_relu_kink_is_not_called_stable_fixed_point(self):
        p = replace(toy_params(degree=0), a=np.zeros((4, 2)), beta=np.zeros((4, 2)),
                    ds=np.full((4, 2), .4))
        transient = fixed_point_diagnostics(toy_circuit(p), np.ones((4, 2)))
        self.assertFalse(transient['is_approximate_fixed_point'])
        self.assertIsNone(transient['linearly_stable_fixed_point'])
        kink = replace(p, poly_coeff=np.tile([0., 1.], (4, 1)), threshold=np.zeros(4))
        kink_report = fixed_point_diagnostics(toy_circuit(kink), np.zeros((4, 2)))
        self.assertTrue(kink_report['is_approximate_fixed_point'])
        self.assertFalse(kink_report['differentiable_at_candidate'])
        self.assertIsNone(kink_report['linearly_stable_fixed_point'])

    def test_jacobian_resource_guard_and_state_validation(self):
        c = toy_circuit()
        h = np.zeros((c.N, c.K))
        with self.assertRaises(ValueError):
            full_state_jacobian(c, h, max_state_dim=h.size-1)
        with self.assertRaises(ValueError):
            fixed_point_residual(c, np.zeros(c.N))
        with self.assertRaises(ValueError):
            constant_input_step(c, h, np.full(c.in_dim, np.nan))

    def test_root_solver_finds_known_fixed_point_and_returns_json_safe_artifact(self):
        p = replace(toy_params(degree=0), a=np.zeros((4, 2)), beta=np.zeros((4, 2)),
                    ds=np.full((4, 2), .4))
        c = toy_circuit(p)
        h = np.arange(8, dtype=float).reshape(4, 2) + 1.
        original_h = h.copy()
        original_circuit = c.state_dict()
        result = find_fixed_point(c, h)
        self.assertTrue(result['solver_success'])
        self.assertTrue(result['success'])
        self.assertTrue(result['is_approximate_fixed_point'])
        self.assertTrue(result['linearly_stable_fixed_point'])
        self.assertGreater(result['jacobian_evaluations'], 0)
        self.assertAlmostEqual(result['full_state_spectral_radius'], .6)
        assert_allclose(result['state'], np.zeros((4, 2)), atol=1e-10)
        assert_array_equal(result['constant_input'], np.zeros(5))
        self.assertFalse(result['uniqueness_claimed'])
        json.dumps(result, allow_nan=False)
        assert_array_equal(h, original_h)
        for key, value in original_circuit.items():
            assert_array_equal(c.state_dict()[key], value)

    def test_root_solver_uses_analytic_residual_jacobian_with_constant_input(self):
        c = toy_circuit()
        h = np.full((c.N, c.K), 20.)
        x = np.array([.1, -.3, .2, .5, 0.])
        result = find_fixed_point(c, h, input_vector=x)
        self.assertTrue(result['success'], result['solver_message'])
        candidate = np.asarray(result['state'])
        self.assertLessEqual(np.max(abs(fixed_point_residual(c, candidate, x))),
                             result['residual_tolerance'])
        assert_array_equal(result['constant_input'], x)
        self.assertEqual(result['solver_jacobian'], 'full_state_jacobian - identity')
        self.assertGreater(result['jacobian_evaluations'], 0)

    def test_solver_termination_alone_never_certifies_transient_success(self):
        c = toy_circuit()
        h = np.full((c.N, c.K), 20.)
        fake = SimpleNamespace(x=h.ravel(), success=True, status=1,
                               message='Deliberately incorrect test termination')

        def pretend_success(fun, flat, jac, method, options):
            self.assertEqual(method, 'hybr')
            # Check the actual callback given to SciPy is J-I, including all
            # recurrent paths, rather than the autonomous map's J alone.
            assert_allclose(jac(flat), full_state_jacobian(c, h)-np.eye(h.size))
            assert_allclose(fun(flat), fixed_point_residual(c, h).ravel())
            return fake

        with patch('scipy.optimize.root', side_effect=pretend_success):
            result = find_fixed_point(c, h)
        self.assertTrue(result['solver_success'])
        self.assertFalse(result['is_approximate_fixed_point'])
        self.assertFalse(result['success'])
        self.assertIsNone(result['linearly_stable_fixed_point'])
        json.dumps(result, allow_nan=False)

    def test_numerical_solver_failure_is_reported_without_false_success(self):
        c = toy_circuit()
        h = np.full((c.N, c.K), 20.)
        with patch('scipy.optimize.root', side_effect=FloatingPointError('test overflow')):
            result = find_fixed_point(c, h)
        self.assertFalse(result['solver_success'])
        self.assertFalse(result['success'])
        self.assertEqual(result['solver_status'], -1)
        self.assertEqual(result['candidate_source'], 'best_finite_evaluated_state')
        assert_array_equal(result['state'], h)
        json.dumps(result, allow_nan=False)
        with self.assertRaises(ValueError):
            find_fixed_point(c, h, max_state_dim=h.size-1)
        with self.assertRaises(ValueError):
            find_fixed_point(c, h, max_evaluations=np.inf)


class NamedRandomnessTests(unittest.TestCase):
    def test_streams_are_stable_namespace_separated_and_order_independent(self):
        left = rng_for(42, 'train').normal(size=40)
        rng_for(42, 'test').normal(size=5000)
        right = rng_for(42, 'train').normal(size=40)
        assert_array_equal(left, right)
        self.assertNotEqual(seed_for(42, 'train'), seed_for(42, 'test'))
        self.assertNotEqual(seed_for(42, 'train'), seed_for(43, 'train'))


class FixedPointExperimentIntegrationTests(unittest.TestCase):
    def test_optional_fixed_point_branch_runs_after_training_and_writes_json(self):
        from grnn_bio.config import ExperimentConfig
        from grnn_bio.experiment import run_suite

        cfg = ExperimentConfig.load(ROOT/'configs'/'smoke.json')
        cfg.name = 'fixed_point_integration_test'
        cfg.conditions = cfg.conditions[:1]
        cfg.training_steps = 1
        cfg.checkpoint_steps = (0, 1)
        cfg.validation_interval = 1
        cfg.run_fixed_points = True
        cfg.run_interventions = False
        cfg.run_psychometrics = False
        cfg.save_trajectories = False
        with tempfile.TemporaryDirectory() as directory:
            summaries = run_suite(cfg, Path(directory)/'run', project_root=ROOT)
            files = list((Path(directory)/'run').rglob('fixed_point.json'))
            self.assertEqual(len(summaries), 1)
            self.assertEqual(len(files), 1)
            result = json.loads(files[0].read_text())
        self.assertEqual(result['state_dimension'], sum(cfg.cell_counts)*8)
        self.assertEqual(np.asarray(result['state']).shape, (sum(cfg.cell_counts), 8))
        self.assertEqual(result['success'],
                         result['solver_success'] and result['is_approximate_fixed_point'])
        assert_array_equal(result['constant_input'], np.zeros(5))
        self.assertFalse(result['global_attractor_claimed'])
        json.dumps(result, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
