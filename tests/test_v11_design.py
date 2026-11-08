"""v11: DMS task, donor-disjoint cohorts and operating-point-matched controls."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grnn_bio.data import load_dataset, partition_cells, sample_cells
from grnn_bio.model import calibrate_current, isolated_response
from grnn_bio.circuit import Circuit
from grnn_bio.config import Condition
from grnn_bio.task import TaskConfig, generate_batch, behavior_metrics


class DmsTaskTest(unittest.TestCase):
    cfg = TaskConfig(task_type='dms', context_timing='late')

    def test_target_is_sample_times_probe_and_balanced(self):
        b = generate_batch(self.cfg, 64, np.random.default_rng(0))
        probe = np.where(b.X[:, :, 2].sum(1) > 0, 1.0, -1.0)
        self.assertTrue(np.all(b.X[:, :, 2:4].sum(axis=(1, 2)) == self.cfg.t_context))
        np.testing.assert_array_equal(b.y, np.sign(b.meta['relevant_coherence']) * probe)
        # Every (sample sign, distractor sign, probe) cell appears equally often.
        cells = {}
        for s, d, p in zip(np.sign(b.meta['motion_coh']), np.sign(b.meta['color_coh']), probe):
            cells[(s, d, p)] = cells.get((s, d, p), 0) + 1
        self.assertEqual(sorted(cells.values()), [8] * 8)

    def test_no_choice_information_before_probe(self):
        b = generate_batch(self.cfg, 64, np.random.default_rng(1))
        pre = b.X[:, :, :].copy()
        pre[b.context_mask | b.decision_mask] = 0.
        # Sample sign and choice are uncorrelated by design.
        self.assertAlmostEqual(float(np.corrcoef(np.sign(b.meta['motion_coh']), b.y)[0, 1]), 0.0, places=12)
        self.assertTrue(np.all(pre[:, :, 2:5] == 0))

    def test_probe_comes_after_delay(self):
        b = generate_batch(self.cfg, 32, np.random.default_rng(2))
        for i in range(b.n_trials):
            probe_t = np.flatnonzero(b.context_mask[i])
            delay_t = np.flatnonzero(b.delay_mask[i])
            decision_t = np.flatnonzero(b.decision_mask[i])
            self.assertGreater(probe_t.min(), delay_t.max())
            self.assertGreater(decision_t.min(), probe_t.max())

    def test_dms_requires_late_timing(self):
        with self.assertRaises(ValueError):
            TaskConfig(task_type='dms', context_timing='early')

    def test_metrics_run(self):
        b = generate_batch(self.cfg, 32, np.random.default_rng(3))
        m = behavior_metrics(np.zeros(b.X.shape[:2]) + 1e-3, b)
        self.assertEqual(m['n_scored'], 32)


class DonorSplitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_dataset(ROOT / 'model/gfr_dataset.json', ROOT / 'data/metadata.csv')

    def test_donor_disjoint_complete_and_balanced(self):
        dev = partition_cells(self.df, 'development', holdout_fraction=.5, seed=1701, unit='donor')
        conf = partition_cells(self.df, 'confirmation', holdout_fraction=.5, seed=1701, unit='donor')
        self.assertEqual(len(dev) + len(conf), len(self.df))
        self.assertFalse(set(dev.donor__id) & set(conf.donor__id))
        self.assertFalse(set(dev.cell_id) & set(conf.cell_id))
        for ct in ('Pyr', 'PV', 'SST', 'VIP'):
            n = (self.df.cell_type == ct).sum()
            self.assertLess(abs((conf.cell_type == ct).sum() - n / 2), 0.05 * n + 2)

    def test_donor_split_is_deterministic(self):
        a = partition_cells(self.df, 'confirmation', holdout_fraction=.5, seed=7, unit='donor')
        b = partition_cells(self.df, 'confirmation', holdout_fraction=.5, seed=7, unit='donor')
        np.testing.assert_array_equal(a.cell_id, b.cell_id)


class MatchedCalibrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = partition_cells(load_dataset(ROOT / 'model/gfr_dataset.json', ROOT / 'data/metadata.csv'), 'development')
        cls.params, rows = sample_cells(df, (20, 5, 3, 3), 11)
        cls.types = rows.cell_type.to_numpy()
        cls.cal = calibrate_current(cls.params)

    def _baseline(self, circuit):
        current = np.broadcast_to(circuit.current_offset, (1, 150, circuit.N))
        return isolated_response(circuit.params, current)[0, 100:].mean(0)

    def test_matched_restores_target_baseline(self):
        for identity in ('within_type_shuffle', 'type_homogeneous', 'cross_type_permutation'):
            ref = Circuit(self.params, self.types, seed=3, identity=identity, calibration=self.cal,
                          calibration_mode='reference')
            matched = Circuit(self.params, self.types, seed=3, identity=identity, calibration=self.cal,
                              calibration_mode='matched')
            target = .05 * matched.params.gamma_hz
            err_matched = np.median(np.abs(self._baseline(matched) - target))
            err_ref = np.median(np.abs(self._baseline(ref) - target))
            self.assertLessEqual(err_matched, err_ref + 1e-9, identity)
            # The manipulation itself is untouched by calibration.
            self.assertEqual(ref.params.fingerprint(), matched.params.fingerprint())

    def test_matched_is_identity_for_intact_bank(self):
        a = Circuit(self.params, self.types, seed=3, calibration=self.cal, calibration_mode='reference')
        b = Circuit(self.params, self.types, seed=3, calibration=self.cal, calibration_mode='matched')
        np.testing.assert_array_equal(a.current_offset, b.current_offset)
        np.testing.assert_array_equal(a.current_scale, b.current_scale)

    def test_condition_rejects_unknown_calibration(self):
        with self.assertRaises(ValueError):
            Condition('x', calibration='other')


if __name__ == '__main__':
    unittest.main()


class OperatingPointTest(unittest.TestCase):
    def test_condition_baseline_fraction_changes_calibration_only(self):
        from grnn_bio.config import ExperimentConfig
        from grnn_bio.experiment import build_circuit
        cfg = ExperimentConfig(cell_counts=(12, 3, 2, 2))
        df = partition_cells(load_dataset(ROOT / 'model/gfr_dataset.json', ROOT / 'data/metadata.csv'), 'development')
        low, _ = build_circuit(cfg, df, Condition('a', 'local', baseline_fraction=.02), 5)
        high, _ = build_circuit(cfg, df, Condition('b', 'local', baseline_fraction=.2), 5)
        default, _ = build_circuit(cfg, df, Condition('c', 'local'), 5)
        self.assertEqual(low.params.fingerprint(), high.params.fingerprint())
        np.testing.assert_array_equal(low.W_abs, high.W_abs)
        self.assertTrue(np.all(high.current_offset >= low.current_offset))
        self.assertGreater(high.current_offset.sum(), low.current_offset.sum())
        np.testing.assert_allclose(default.calibration_info['target_rate_hz'], .05 * default.gamma_hz)
        # Matched recalibration of a manipulated bank uses the condition's fraction too.
        shuf, _ = build_circuit(cfg, df, Condition('d', 'local', 'within_type_shuffle', baseline_fraction=.2), 5)
        np.testing.assert_allclose(shuf.calibration_info['target_rate_hz'], .2 * shuf.gamma_hz)
