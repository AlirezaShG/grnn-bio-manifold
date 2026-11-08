"""Hand-calculated factorial contrasts and validation-only tuning contracts."""
from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from grnn_bio.config import Condition, ExperimentConfig, condition_plan
from grnn_bio.design import factorial_contrasts
from grnn_bio.cli import main


# Values are written explicitly to avoid reproducing the implementation's
# signed-contrast formula inside the expected-value calculation.
HAND_TABLE = {
    (0, 0, 0): .20, (0, 0, 1): .17,
    (0, 1, 0): .24, (0, 1, 1): .27,
    (1, 0, 0): .30, (1, 0, 1): .25,
    (1, 1, 0): .42, (1, 1, 1): .39,
}
EXPECTED_EFFECTS = {
    'identity': .12,
    'topology': .10,
    'learner': -.02,
    'identity x topology': .06,
    'identity x learner': -.04,
    'topology x learner': .04,
    'identity x topology x learner': -.04,
}


def factorial_rows(seeds=(10, 20, 30)):
    rows = []
    for offset, seed in enumerate(seeds):
        for (a, b, c), value in HAND_TABLE.items():
            rows.append({
                'seed': seed,
                'identity': ('within_type_shuffle', 'authentic')[a],
                'topology': ('degree_rewired', 'type_prior')[b],
                'learner': ('bptt', 'local')[c],
                'accuracy_incongruent': value + .01 * offset,
                'training_noise_std': 0.,
                'split': 'test_delay',
            })
    return rows


class FactorialContrastTests(unittest.TestCase):
    def test_hand_calculated_main_did_and_ddd_effects(self):
        result = factorial_contrasts(factorial_rows())
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(len(result['contrasts']), 7)
        self.assertEqual(result['positive_coding'], {
            'identity': 'authentic', 'topology': 'type_prior', 'learner': 'local'})
        effects = {r['term']: r for r in result['contrasts']}
        for term, expected in EXPECTED_EFFECTS.items():
            with self.subTest(term=term):
                row = effects[term]
                # For example AB DID is .08 at C=0 and .04 at C=1:
                # its average is .06, whereas the DDD is .04-.08=-.04.
                np.testing.assert_allclose(row['per_network_contrast'], [expected] * 3,
                                           atol=1e-14, rtol=0)
                self.assertAlmostEqual(row['mean_difference_a_minus_b'], expected, places=13)
                self.assertEqual(row['family_size'], 7)
                self.assertGreaterEqual(row['p_holm'], row['p_value'])
                self.assertLessEqual(row['p_holm'], 1.)

    def test_reordering_rows_and_seed_specific_intercepts_preserve_effects(self):
        rows = factorial_rows(seeds=(19, 7, 101))
        before = factorial_contrasts(rows)
        np.random.default_rng(42).shuffle(rows)
        after = factorial_contrasts(rows)
        for a, b in zip(before['contrasts'], after['contrasts']):
            self.assertEqual(a['network_seeds'], [7, 19, 101])
            np.testing.assert_allclose(a['per_network_contrast'], b['per_network_contrast'], atol=1e-14)
            self.assertAlmostEqual(a['p_value'], b['p_value'], places=14)

    def test_incomplete_seed_is_not_silently_dropped(self):
        rows = factorial_rows()
        rows.pop()
        result = factorial_contrasts(rows)
        self.assertEqual(result['status'], 'not_estimable_incomplete_factorial')
        self.assertEqual(result['contrasts'], [])
        self.assertEqual(result['available_cells_per_seed'], {'10': 8, '20': 8, '30': 7})

    def test_empty_design_is_explicitly_not_estimable(self):
        result = factorial_contrasts([])
        self.assertEqual(result['status'], 'not_estimable_incomplete_factorial')
        self.assertEqual(result['contrasts'], [])

    def test_duplicate_cell_per_network_is_rejected(self):
        rows = factorial_rows()
        rows.append(copy.deepcopy(rows[0]))
        with self.assertRaises(ValueError):
            factorial_contrasts(rows)

    def test_invalid_pairing_seed_ids_are_rejected(self):
        for invalid in (1.5, -1, True, '10', np.nan):
            with self.subTest(seed=invalid):
                rows = factorial_rows(seeds=(10,))
                for row in rows:
                    row['seed'] = invalid
                with self.assertRaises(ValueError):
                    factorial_contrasts(rows)

    def test_mixed_endpoints_are_rejected_even_for_disjoint_seed_ids(self):
        rows = factorial_rows(seeds=(10, 20))
        for row in rows:
            if row['seed'] == 20:
                row['split'] = 'test_id'
        with self.assertRaises(ValueError):
            factorial_contrasts(rows)

    def test_nonfinite_outcome_is_rejected(self):
        for invalid in (np.nan, np.inf, -np.inf):
            rows = factorial_rows()
            rows[0]['accuracy_incongruent'] = invalid
            with self.assertRaises(ValueError):
                factorial_contrasts(rows)

    def test_noisy_and_nonfactorial_controls_do_not_contaminate_primary_family(self):
        rows = factorial_rows()
        noisy_duplicate = {**rows[0], 'training_noise_std': .1, 'accuracy_incongruent': .99}
        reservoir = {**rows[0], 'learner': 'readout_only', 'accuracy_incongruent': .01}
        result = factorial_contrasts(rows + [noisy_duplicate, reservoir])
        self.assertEqual(len(result['contrasts']), 7)
        for row in result['contrasts']:
            self.assertAlmostEqual(row['mean_difference_a_minus_b'], EXPECTED_EFFECTS[row['term']], places=13)

    def test_single_pair_reports_effect_without_inferential_p_value(self):
        result = factorial_contrasts(factorial_rows(seeds=(1,)))
        self.assertEqual(result['status'], 'complete')
        for row in result['contrasts']:
            self.assertEqual(row['n_pairs'], 1)
            self.assertIsNone(row['p_value'])
            self.assertIsNone(row['p_holm'])
            self.assertIsNone(row['ci_low'])


class TuningContractTests(unittest.TestCase):
    def _invoke(self, cfg, fake_run, parent):
        path = parent / 'config.json'
        path.write_text(json.dumps(cfg.as_dict()), encoding='utf-8')
        output = parent / 'tuning'
        with patch('grnn_bio.experiment.run_suite', side_effect=fake_run) as mocked:
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(['tune', '--config', str(path), '--output', str(output)])
        return code, output, mocked

    def test_equal_distinct_candidate_budgets_with_validation_only_selection(self):
        cfg = ExperimentConfig(conditions=condition_plan('full'))
        calls = []

        def fake_run(candidate_cfg, output, **kwargs):
            calls.append((copy.deepcopy(candidate_cfg), Path(output), kwargs))
            index = int(Path(output).name.split('_')[-1])
            values = [.4, .2, .3]
            return [{'seed': s, 'best_validation_bce': values[index] + i * .001}
                    for i, s in enumerate(candidate_cfg.seeds)]

        with tempfile.TemporaryDirectory() as temp:
            code, output, _ = self._invoke(cfg, fake_run, Path(temp))
            self.assertEqual(code, 0)
            selected = json.loads((output / 'selected_development_config.json').read_text())
            summary = json.loads((output / 'tuning_summary.json').read_text())
            record = json.loads((output / 'selection_record.json').read_text())
        learners = {'bptt', 'local', 'local_no_beta', 'readout_only'}
        self.assertEqual(len(calls), 3 * len(learners))
        self.assertEqual(set(selected['learning_rates']), learners)
        self.assertEqual(selected['conditions'], cfg.as_dict()['conditions'])
        self.assertFalse(record['test_evaluated'])
        self.assertEqual(record['candidates_per_learner'], 3)
        for learner in learners:
            subset = [(run_cfg, kwargs) for run_cfg, _, kwargs in calls
                      if run_cfg.conditions[0].learner == learner]
            self.assertEqual(len(subset), 3)
            self.assertEqual(len({c.learning_rates[learner]['lr_out'] for c, _ in subset}), 3)
            self.assertEqual(selected['learning_rates'][learner], {'lr_rec': .001, 'lr_out': .03})
            for run_cfg, kwargs in subset:
                self.assertFalse(kwargs['test'])
                self.assertEqual(tuple(run_cfg.seeds), tuple(cfg.seeds))
                self.assertEqual(run_cfg.training_steps, cfg.training_steps)
                self.assertEqual(run_cfg.batch_size, cfg.batch_size)
                self.assertEqual(run_cfg.cell_cohort, 'development')
                self.assertEqual(len(run_cfg.conditions), 1)
                self.assertEqual(run_cfg.conditions[0].training_noise_std, 0.)
        self.assertTrue(all(not row['test_evaluated'] for row in summary))

    def test_ties_use_first_declared_candidate_without_testing(self):
        cfg = ExperimentConfig(conditions=[Condition('reference')])
        with tempfile.TemporaryDirectory() as temp:
            def fake_run(c, output, **kwargs):
                self.assertFalse(kwargs['test'])
                return [{'seed': s, 'best_validation_bce': .5} for s in c.seeds]
            _, output, _ = self._invoke(cfg, fake_run, Path(temp))
            selected = json.loads((output / 'selected_development_config.json').read_text())
        self.assertEqual(selected['learning_rates']['bptt'], {'lr_rec': .0003, 'lr_out': .01})

    def test_confirmation_tuning_is_rejected_without_any_run(self):
        cfg = ExperimentConfig(stage='confirmation', cell_cohort='confirmation')
        with tempfile.TemporaryDirectory() as temp:
            calls = []
            def fake_run(*args, **kwargs):
                calls.append((args, kwargs))
                return []
            with self.assertRaises(ValueError):
                self._invoke(cfg, fake_run, Path(temp))
            self.assertEqual(calls, [])

    def test_ambiguous_feedback_representatives_are_rejected(self):
        cfg = ExperimentConfig(conditions=[
            Condition('random_local', 'local', feedback_mode='random'),
            Condition('symmetric_local', 'local', feedback_mode='symmetric'),
        ])
        with tempfile.TemporaryDirectory() as temp:
            def fake_run(*args, **kwargs):
                self.fail('No training is permitted for ambiguous tuning representatives')
            with self.assertRaises(ValueError):
                self._invoke(cfg, fake_run, Path(temp))

    def test_missing_or_nonfinite_candidate_scores_are_rejected(self):
        cfg = ExperimentConfig(conditions=[Condition('reference')])
        for scores in ([], [.5], [.5, np.nan, .5], [.5, np.inf, .5]):
            with self.subTest(scores=scores), tempfile.TemporaryDirectory() as temp:
                def fake_run(*args, **kwargs):
                    return [{'best_validation_bce': score} for score in scores]
                with self.assertRaises(ValueError):
                    self._invoke(cfg, fake_run, Path(temp))

    def test_noisy_conditions_are_not_silently_tuned_as_noiseless(self):
        cfg = ExperimentConfig(conditions=[Condition('noisy_bptt', training_noise_std=.1)])
        with tempfile.TemporaryDirectory() as temp:
            def fake_run(*args, **kwargs):
                self.fail('Noisy-only configuration must request its own declared tuning design')
            with self.assertRaises(ValueError):
                self._invoke(cfg, fake_run, Path(temp))


if __name__ == '__main__':
    unittest.main()
