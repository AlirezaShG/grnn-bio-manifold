"""Scientific-validity checks for trial design, holdouts and paired inference."""

from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np

from grnn_bio.task import TaskConfig, generate_batch, behavior_metrics, decision_values
from grnn_bio.analysis import analyze_population, fit_population_analysis, phase_average, perturbation_recovery, temporal_generalization
from grnn_bio.statistics import paired_comparison, holm_adjust, paired_condition_table, network_level_cell_association
from grnn_bio.interventions import make_rate_multiplier, evaluate_rate_interventions, evaluate_pulse_recovery


class TaskValidityTests(unittest.TestCase):
    def test_factorial_balance_and_unbiased_partial_block(self):
        cfg = TaskConfig()
        for n in (8, 64, 67):
            batch = generate_batch(cfg, n, np.random.default_rng(7))
            cells = np.column_stack([batch.meta["context_rule"], np.sign(batch.meta["relevant_coherence"]), np.sign(batch.meta["irrelevant_coherence"])])
            _, counts = np.unique(cells, axis=0, return_counts=True)
            self.assertEqual(len(counts), 8)
            self.assertLessEqual(counts.max() - counts.min(), 1)
            if n % 8 == 0:
                self.assertEqual(np.sum(batch.y > 0), n // 2)
                self.assertEqual(np.sum(batch.meta["is_congruent"]), n // 2)

    def test_variable_delay_masks_go_and_padding(self):
        cfg = TaskConfig(delay_bins=(3, 5, 9))
        batch = generate_batch(cfg, 128, np.random.default_rng(11))
        np.testing.assert_array_equal(batch.delay_mask.sum(1), batch.meta["delay_bins"])
        np.testing.assert_array_equal(batch.decision_mask.sum(1), cfg.t_decision)
        np.testing.assert_array_equal(batch.X[:, :, 4], batch.decision_mask)
        self.assertTrue(np.all(batch.X[~batch.valid_mask] == 0))
        self.assertTrue(np.all(batch.X[batch.delay_mask] == 0))
        self.assertEqual(len(np.unique(batch.meta["decision_start"])), 3)

    def test_late_context_requires_stream_storage(self):
        cfg = TaskConfig(context_timing="late", delay_bins=(5,))
        batch = generate_batch(cfg, 32, np.random.default_rng(1))
        self.assertTrue(np.all(batch.X[batch.stimulus_mask, 2:4] == 0))
        self.assertTrue(np.all(batch.X[batch.delay_mask] == 0))
        np.testing.assert_array_equal(batch.context_mask.sum(1), cfg.t_context)
        np.testing.assert_array_equal(batch.X[:, :, 2] + batch.X[:, :, 3], batch.context_mask)

    def test_named_rng_external_reproducibility(self):
        a = generate_batch(TaskConfig(), 64, np.random.default_rng(33))
        b = generate_batch(TaskConfig(), 64, np.random.default_rng(33))
        c = generate_batch(TaskConfig(), 64, np.random.default_rng(34))
        np.testing.assert_array_equal(a.X, b.X)
        self.assertFalse(np.array_equal(a.X, c.X))

    def test_zero_coherence_is_not_forced_positive_accuracy(self):
        batch = generate_batch(TaskConfig(), 64, np.random.default_rng(4), relevant_coherence=0)
        self.assertFalse(batch.meta["valid_choice"].any())
        self.assertTrue(np.all(batch.y == 0))
        metrics = behavior_metrics(np.ones(batch.X.shape[:2]), batch)
        self.assertEqual(metrics["n_scored"], 0)
        self.assertTrue(np.isnan(metrics["accuracy"]))
        self.assertEqual(metrics["choice_probability_plus"], 1.0)

    def test_readout_uses_per_trial_go_not_batch_last_window(self):
        batch = generate_batch(TaskConfig(delay_bins=(2, 20)), 64, np.random.default_rng(44))
        logits = np.full(batch.X.shape[:2], -999.0)
        logits[batch.decision_mask] = np.broadcast_to(batch.y[:, None], logits.shape)[batch.decision_mask]
        np.testing.assert_array_equal(decision_values(logits, batch), batch.y)
        self.assertEqual(behavior_metrics(logits, batch)["accuracy"], 1.0)

    def test_distractor_does_not_change_target_or_timing(self):
        cfg = TaskConfig()
        clean = generate_batch(cfg, 64, np.random.default_rng(71), distractor_std=0.0)
        noisy = generate_batch(cfg, 64, np.random.default_rng(71), distractor_std=0.7)
        np.testing.assert_array_equal(clean.y, noisy.y)
        np.testing.assert_array_equal(clean.decision_mask, noisy.decision_mask)
        np.testing.assert_array_equal(clean.X[:, :, 2:], noisy.X[:, :, 2:])
        self.assertTrue(np.all(noisy.distractor_mask <= noisy.delay_mask))
        self.assertTrue(np.any(noisy.X[noisy.distractor_mask, :2] != 0))

    def test_invalid_task_definitions_fail(self):
        for kwargs in ({"delay_bins": (0,)}, {"noise_std": -1}, {"coherence_levels": (0.0,)}, {"context_timing": "undefined"}):
            with self.assertRaises(ValueError):
                TaskConfig(**kwargs)


class HeldoutAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.cfg = TaskConfig(t_fixation=1, t_stimulus=2, t_decision=2, delay_bins=(5, 7))
        self.fit_batch = generate_batch(self.cfg, 160, np.random.default_rng(101))
        self.eval_batch = generate_batch(self.cfg, 160, np.random.default_rng(102))

    def test_phase_average_ignores_padding_and_other_epochs(self):
        rates = np.full((*self.fit_batch.X.shape[:2], 3), 1000.0)
        rates[self.fit_batch.delay_mask] = 7.0
        np.testing.assert_allclose(phase_average(rates, self.fit_batch), 7.0)

    def test_independent_choice_signal_decodes_and_reversal_stays_negative(self):
        rng = np.random.default_rng(45)
        fit = rng.normal(0, 0.1, (*self.fit_batch.X.shape[:2], 6))
        evaluation = rng.normal(0, 0.1, (*self.eval_batch.X.shape[:2], 6))
        fit[:, :, 0] += self.fit_batch.y[:, None]
        evaluation[:, :, 0] -= self.eval_batch.y[:, None]
        metrics = analyze_population(fit, self.fit_batch, evaluation, self.eval_batch)
        self.assertLess(metrics["choice_dprime_heldout"], -1.0)
        self.assertLess(metrics["choice_decoder_accuracy"], 0.1)

    def test_null_holdout_does_not_inherit_high_dimensional_fit_separation(self):
        rng = np.random.default_rng(90)
        fit = rng.normal(size=(*self.fit_batch.X.shape[:2], 80))
        evaluation = rng.normal(size=(*self.eval_batch.X.shape[:2], 80))
        metrics = analyze_population(fit, self.fit_batch, evaluation, self.eval_batch)
        self.assertGreater(metrics["choice_decoder_accuracy"], 0.35)
        self.assertLess(metrics["choice_decoder_accuracy"], 0.65)
        self.assertLess(abs(metrics["choice_dprime_heldout"]), 0.5)

    def test_fitted_pca_and_scaling_do_not_change_on_evaluation(self):
        rng = np.random.default_rng(63)
        rates = rng.normal(size=(*self.fit_batch.X.shape[:2], 4))
        fit = fit_population_analysis(rates, self.fit_batch)
        original_mean = fit.pca_mean.copy()
        original_scale = fit.choice_probe.scale.copy()
        ev = rng.normal(size=(*self.eval_batch.X.shape[:2], 4)) + 1000
        fit.evaluate(ev, self.eval_batch)
        np.testing.assert_array_equal(fit.pca_mean, original_mean)
        np.testing.assert_array_equal(fit.choice_probe.scale, original_scale)

    def test_reusing_identical_batch_for_fitting_and_testing_is_rejected(self):
        rates = np.random.default_rng(1).normal(size=(*self.fit_batch.X.shape[:2], 4))
        with self.assertRaises(ValueError):
            analyze_population(rates, self.fit_batch, rates, self.fit_batch)

    def test_temporal_generalization_uses_heldout_labels_only_for_scoring(self):
        rng = np.random.default_rng(3)
        fit = rng.normal(0, 0.05, (*self.fit_batch.X.shape[:2], 3))
        ev = rng.normal(0, 0.05, (*self.eval_batch.X.shape[:2], 3))
        fit[:, :, 0] += self.fit_batch.y[:, None]
        ev[:, :, 0] += self.eval_batch.y[:, None]
        matrix = temporal_generalization(fit, self.fit_batch, ev, self.eval_batch, n_bins=3)
        np.testing.assert_allclose(matrix, 1.0)

    def test_expansion_is_retained_and_zero_displacement_is_undefined(self):
        a = np.zeros((2, 4, 1))
        b = a.copy()
        b[:, 1, :] = 1.0
        b[:, 3, :] = 2.0
        pulse = np.zeros((2, 4), bool)
        endpoint = pulse.copy()
        pulse[:, 1] = True
        endpoint[:, 3] = True
        metric = perturbation_recovery(a, b, pulse, endpoint)
        self.assertEqual(metric["endpoint_to_initial_ratio"], 2.0)
        self.assertEqual(metric["recovery_fraction_unclipped"], -1.0)
        zero = perturbation_recovery(a, a, pulse, endpoint)
        self.assertTrue(np.isnan(zero["endpoint_to_initial_ratio"]))
        self.assertEqual(zero["n_zero_initial_displacement"], 2)


class StatisticsValidityTests(unittest.TestCase):
    def test_exact_six_pair_resolution(self):
        result = paired_comparison(np.arange(6.0) + 1, np.arange(6.0), seed=1, n_resamples=200)
        self.assertEqual(result["p_value"], 2 / 64)
        self.assertEqual(result["mean_difference_a_minus_b"], 1.0)
        self.assertEqual(result["test_method"], "exact_paired_sign_flip_two_sided")

    def test_no_claim_of_inference_from_one_seed(self):
        result = paired_comparison([0.9], [0.5], n_resamples=100)
        self.assertIsNone(result["p_value"])
        self.assertIsNone(result["ci_low"])
        self.assertTrue(result["small_sample"])

    def test_identical_pairs_have_null_p_and_zero_ci(self):
        result = paired_comparison([1, 2, 3], [1, 2, 3], n_resamples=100)
        self.assertEqual(result["p_value"], 1.0)
        self.assertEqual(result["ci_low"], 0.0)

    def test_holm_known_values_and_missing_family_member(self):
        np.testing.assert_allclose(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        self.assertEqual(holm_adjust([0.01, None]), [0.02, None])

    def test_pairing_by_id_and_incomplete_pairs_rejected(self):
        rows = [{"condition": "A", "seed": 2, "acc": 0.2}, {"condition": "A", "seed": 1, "acc": 0.3}, {"condition": "B", "seed": 1, "acc": 0.4}, {"condition": "B", "seed": 2, "acc": 0.3}]
        result = paired_condition_table(rows, "A", "acc", n_resamples=100)[0]
        np.testing.assert_allclose(result["differences"], [0.1, 0.1])
        with self.assertRaises(ValueError):
            paired_condition_table(rows[:-1], "A", "acc", n_resamples=100)
        with self.assertRaises(ValueError):
            paired_condition_table(rows + rows[:1], "A", "acc", n_resamples=100)

    def test_nested_cell_association_uses_networks_and_unique_cell_ids(self):
        rows = [{"seed": seed, "cell_id": cell, "x": cell, "y": 2 * cell + seed} for seed in (1, 2, 3) for cell in range(5)]
        rows += [rows[0].copy(), rows[0].copy()]
        result = network_level_cell_association(rows, "x", "y")
        self.assertEqual(result["n_networks_valid"], 3)
        self.assertEqual(result["per_network"][0]["n_unique_cells"], 5)
        self.assertIsNone(result["pooled_cell_p_value"])


class _FakeCircuit:
    N = 5
    cell_types = np.array(["Pyr", "Pyr", "PV", "SST", "VIP"])
    current_scale = np.array([20.0, 40.0, 60.0, 80.0, 100.0])

    def __init__(self):
        self.noise_seen = []
        self.multipliers_seen = []

    def forward(self, X, *, current_noise=None, rate_multiplier=None, record_states=False):
        self.noise_seen.append(current_noise)
        self.multipliers_seen.append(rate_multiplier)
        rates = np.ones((*X.shape[:2], self.N))
        if current_noise is not None:
            rates += 0.01 * current_noise
        if rate_multiplier is not None:
            rates *= rate_multiplier
        return SimpleNamespace(rates_hz=rates, rates_norm=rates, logits=rates[:, :, 0] - 0.5, states=rates[:, :, :, None] if record_states else None)


class InterventionValidityTests(unittest.TestCase):
    def setUp(self):
        self.batch = generate_batch(TaskConfig(delay_bins=(4, 6)), 16, np.random.default_rng(5))

    def test_phase_specific_clamp_leaves_other_times_and_cells_intact(self):
        mult = make_rate_multiplier(self.batch, 5, [2], "delay", 1.0)
        self.assertTrue(np.all(mult[:, :, 2][self.batch.delay_mask] == 0))
        self.assertTrue(np.all(mult[:, :, 2][~self.batch.delay_mask] == 1))
        self.assertTrue(np.all(mult[:, :, [0, 1, 3, 4]] == 1))

    def test_zero_dose_and_matched_controls_share_noise(self):
        circuit = _FakeCircuit()
        noise = np.random.default_rng(5).normal(size=(*self.batch.X.shape[:2], 5))
        rows, trial_rows = evaluate_rate_interventions(circuit, self.batch, current_noise=noise, phases=("delay",), doses=(0.0, 1.0), target_types=("PV",), control_seed=2)
        self.assertEqual(len(rows), 5)
        self.assertEqual(len(trial_rows), self.batch.n_trials * len(rows))
        self.assertTrue(all(n is noise for n in circuit.noise_seen))
        zero_dose = [row for row in rows if row["dose"] == 0]
        self.assertTrue(all(row["delta_accuracy"] == 0 for row in zero_dose))
        controls = [row for row in rows if row.get("intervention_kind") == "matched_size_control"]
        self.assertTrue(all(row["n_targeted"] == 1 for row in controls))
        self.assertTrue(all(2 not in row["target_indices"] for row in controls))

    def test_current_pulse_reports_hidden_state_and_does_not_claim_attractor(self):
        circuit = _FakeCircuit()
        result = evaluate_pulse_recovery(circuit, self.batch, pulse_std_pa=5, pulse_seed=1)
        self.assertIn("hidden_state_displacement", result)
        self.assertFalse(result["attractor_claim_supported"])
        self.assertEqual(result["rate_displacement"]["recovery_fraction_unclipped"], 1.0)
        physical_difference = (circuit.noise_seen[1] - circuit.noise_seen[0]) * circuit.current_scale
        expected_direction = np.random.default_rng(1).normal(0, 5, size=(self.batch.n_trials, circuit.N))
        np.testing.assert_allclose(physical_difference[np.arange(self.batch.n_trials), result["pulse_steps"]], expected_direction)


if __name__ == "__main__":
    unittest.main()
