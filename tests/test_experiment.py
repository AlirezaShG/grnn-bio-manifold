"""End-to-end experiments on the bundled real K=8 cells, with tiny budgets.

These tests check data access boundaries, paired random streams, saved outputs,
checkpoint selection and interrupted resume. Their accuracies are not research
results. Temporary outputs and a frozen source/data snapshot are discarded.
"""

from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from grnn_bio import experiment
from grnn_bio.cli import main as cli_main
from grnn_bio.config import Condition, ExperimentConfig
from grnn_bio.data import load_dataset, partition_cells
from grnn_bio.io import write_json
from grnn_bio.task import TaskConfig


ROOT = Path(__file__).resolve().parents[1]


class ExperimentIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = partition_cells(load_dataset(ROOT / "model/gfr_dataset.json", ROOT / "data/metadata.csv"), "development")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name)
        self.project = self.workspace / "project"
        self.project.mkdir()
        shutil.copytree(ROOT / "grnn_bio", self.project / "grnn_bio", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for relative in ("model/gfr_dataset.json", "data/metadata.csv"):
            path = self.project / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, path)

    def config(self, **kwargs):
        base = dict(
            name="integration_test",
            seeds=(7,),
            cell_counts=(4, 2, 1, 1),
            task=TaskConfig(t_fixation=1, t_stimulus=2, t_decision=1, delay_bins=(5, 6)),
            conditions=[Condition("bptt")],
            training_steps=4,
            batch_size=8,
            validation_interval=2,
            validation_trials=8,
            test_trials=8,
            analysis_trials=8,
            checkpoint_steps=(0, 2, 4),
            run_interventions=False,
            run_psychometrics=False,
            save_trajectories=False,
            ood_delays=(8, 10),
        )
        base.update(kwargs)
        return ExperimentConfig(**base).validate()

    def run_one(self, cfg, name, *, condition=None, test=False, resume=False):
        return experiment.run_one(cfg, self.df, condition or cfg.conditions[0], cfg.seeds[0], self.workspace / name, project_root=self.project, resume=resume, test=test)

    def assert_checkpoint_arrays_identical(self, path_a, path_b):
        with np.load(path_a, allow_pickle=False) as a, np.load(path_b, allow_pickle=False) as b:
            self.assertEqual(set(a.files), set(b.files))
            for key in a.files:
                if key != "metadata_json":
                    np.testing.assert_array_equal(a[key], b[key], err_msg=key)
            meta_a, meta_b = json.loads(str(a["metadata_json"])), json.loads(str(b["metadata_json"]))
            self.assertEqual(meta_a["optimizer"], meta_b["optimizer"])
            for key in ("step", "best_step", "best_validation_bce", "simulated_trial_bins"):
                self.assertEqual(meta_a["run"][key], meta_b["run"][key])

    def test_test_false_cannot_enter_heldout_or_mechanistic_branches(self):
        cfg = self.config(run_interventions=True, run_psychometrics=True, save_trajectories=True)
        with mock.patch.object(experiment, "_test_batches", side_effect=AssertionError("test generator accessed")), mock.patch.object(experiment, "fit_population_analysis", side_effect=AssertionError("analysis accessed")), mock.patch.object(experiment, "evaluate_rate_interventions", side_effect=AssertionError("interventions accessed")):
            summary = self.run_one(cfg, "validation_only", test=False)
        self.assertFalse(summary["test_enabled"])
        self.assertEqual(summary["metrics"], [])
        directory = self.workspace / "validation_only"
        self.assertTrue((directory / "best_checkpoint.npz").exists())
        self.assertFalse((directory / "trial_predictions.csv").exists())
        self.assertFalse((directory / "population_analysis.json").exists())
        self.assertFalse((directory / "psychometrics.csv").exists())

    def test_complete_tiny_run_writes_five_splits_and_consistent_checkpoints(self):
        cfg = self.config()
        summary = self.run_one(cfg, "complete", test=True)
        self.assertEqual({x["split"] for x in summary["metrics"]}, {"test_id", "test_delay", "test_sensory_noise", "test_distractor", "test_combined"})
        directory = self.workspace / "complete"
        trials = pd.read_csv(directory / "trial_predictions.csv")
        self.assertEqual(len(trials), 5 * cfg.test_trials)
        self.assertEqual(trials.groupby("split").trial_id.nunique().to_dict(), {name: cfg.test_trials for name in trials.split.unique()})
        for row in summary["metrics"]:
            observed = trials[trials.split == row["split"]]
            self.assertAlmostEqual(observed.correct.mean(), row["accuracy"])
            self.assertAlmostEqual(observed.bce.mean(), row["bce"])
        with np.load(directory / "best_checkpoint.npz", allow_pickle=False) as checkpoint:
            metadata = json.loads(str(checkpoint["metadata_json"]))
            self.assertEqual(metadata["run"]["step"], summary["best_step"])
            self.assertAlmostEqual(metadata["run"]["best_validation_bce"], summary["best_validation_bce"])
        manifest = json.loads((directory / "manifest.json").read_text())
        self.assertEqual(set(manifest["input_hashes"]), {"model/gfr_dataset.json", "data/metadata.csv"})
        self.assertFalse(manifest["neuron_parameters_trainable"])
        geometry = json.loads((directory / "checkpoint_geometry.json").read_text())
        self.assertEqual([x["step"] for x in geometry], list(cfg.checkpoint_steps))

    def test_analysis_handles_short_delays_and_paired_configured_noise(self):
        cfg = self.config(task=TaskConfig(t_fixation=1, t_stimulus=2, t_decision=1, delay_bins=(2, 3)), training_steps=2, checkpoint_steps=(0, 1, 2), validation_interval=1, evaluation_noise_std=0.05, run_interventions=True, intervention_doses=(0.0, 1.0), run_psychometrics=True)
        summary = self.run_one(cfg, "short", test=True)
        self.assertEqual(len(summary["metrics"]), 5)
        directory = self.workspace / "short"
        matrix = np.asarray(json.loads((directory / "temporal_generalization.json").read_text()))
        self.assertEqual(matrix.shape, (2, 2))
        interventions = pd.read_csv(directory / "rate_interventions.csv")
        zero = interventions[interventions.dose == 0]
        np.testing.assert_allclose(zero.delta_accuracy, 0)
        self.assertTrue(interventions.background_noise_shared.all())
        psych = pd.read_csv(directory / "psychometrics.csv")
        self.assertEqual(set(psych.context), {0, 1})
        self.assertTrue((psych[psych.coherence == 0].n_scored == 0).all())
        self.assertTrue(psych[psych.coherence == 0].accuracy.isna().all())
        pulse = json.loads((directory / "pulse_recovery.json").read_text())
        self.assertFalse(pulse["attractor_claim_supported"])
        self.assertEqual(pulse["background_noise_units"], "dimensionless_drive")

    def test_validation_frequency_and_learning_rule_do_not_change_training_data(self):
        bptt, local = Condition("bptt"), Condition("local", "local")
        cfg = self.config(conditions=[bptt, local])
        recorded = {}
        original = experiment._gradient_step
        for label, condition, interval in (("bptt", bptt, 2), ("local", local, 2), ("frequent_validation", bptt, 1)):
            recorded[label] = []

            def capture(circuit, batch, cond, settings, seed, step, *, bucket=recorded[label]):
                bucket.append((batch.X.copy(), batch.y.copy(), batch.decision_mask.copy()))
                return original(circuit, batch, cond, settings, seed, step)

            settings = replace(cfg, validation_interval=interval)
            with mock.patch.object(experiment, "_gradient_step", side_effect=capture):
                self.run_one(settings, label, condition=condition, test=False)
        for label in ("local", "frequent_validation"):
            for a, b in zip(recorded["bptt"], recorded[label]):
                for x, y in zip(a, b):
                    np.testing.assert_array_equal(x, y)

    def test_paired_conditions_reuse_identical_test_trials(self):
        a, b = Condition("bptt"), Condition("local", "local")
        cfg = self.config(conditions=[a, b], training_steps=2, checkpoint_steps=(0, 2))
        self.run_one(cfg, "pair_a", condition=a, test=True)
        self.run_one(cfg, "pair_b", condition=b, test=True)
        columns = ["split", "trial_id", "target", "context", "congruent", "delay_bins", "relevant_coherence", "irrelevant_coherence"]
        left = pd.read_csv(self.workspace / "pair_a/trial_predictions.csv")[columns]
        right = pd.read_csv(self.workspace / "pair_b/trial_predictions.csv")[columns]
        pd.testing.assert_frame_equal(left, right)

    def test_interrupted_resume_is_numerically_identical_for_bptt_and_noisy_learning(self):
        original = experiment._gradient_step
        for condition in (Condition("bptt"), Condition("perturb", "node_perturbation", training_noise_std=0.1)):
            with self.subTest(learner=condition.learner):
                cfg = self.config(conditions=[condition])
                interrupted_name, uninterrupted_name = condition.name + "_interrupted", condition.name + "_uninterrupted"

                def interrupt(circuit, batch, cond, settings, seed, step):
                    if step == 3:
                        raise RuntimeError("deliberate interruption after saved step 2")
                    return original(circuit, batch, cond, settings, seed, step)

                with mock.patch.object(experiment, "_gradient_step", side_effect=interrupt):
                    with self.assertRaisesRegex(RuntimeError, "deliberate interruption"):
                        self.run_one(cfg, interrupted_name, test=False)
                resumed = self.run_one(cfg, interrupted_name, test=False, resume=True)
                uninterrupted = self.run_one(cfg, uninterrupted_name, test=False)
                self.assertEqual(resumed["best_step"], uninterrupted["best_step"])
                self.assertEqual(resumed["best_validation_bce"], uninterrupted["best_validation_bce"])
                self.assertEqual(resumed["simulated_trial_bins"], uninterrupted["simulated_trial_bins"])
                for checkpoint in ("best_checkpoint.npz", "last_checkpoint.npz"):
                    self.assert_checkpoint_arrays_identical(self.workspace / interrupted_name / checkpoint, self.workspace / uninterrupted_name / checkpoint)

    def test_completed_resume_returns_without_retraining_and_rejects_config_changes(self):
        cfg = self.config()
        first = self.run_one(cfg, "resume_complete", test=False)
        with mock.patch.object(experiment, "_gradient_step", side_effect=AssertionError("retrained completed run")):
            second = self.run_one(cfg, "resume_complete", test=False, resume=True)
        self.assertEqual(first, second)
        with self.assertRaisesRegex(ValueError, "different"):
            self.run_one(replace(cfg, lr_out=cfg.lr_out * 2), "resume_complete", test=False, resume=True)

    def test_data_bytes_are_part_of_resume_identity(self):
        cfg = self.config()
        self.run_one(cfg, "data_bound", test=False)
        dataset = self.project / "model/gfr_dataset.json"
        dataset.write_bytes(dataset.read_bytes() + b"\n")
        # Even semantically unchanged bytes must change the saved provenance.
        with self.assertRaisesRegex(ValueError, "different"):
            self.run_one(cfg, "data_bound", test=False, resume=True)

    def test_tuning_command_passes_test_false_and_selects_validation_only(self):
        cfg = self.config(conditions=[Condition("bptt"), Condition("local", "local")])
        config_file = self.workspace / "tune_config.json"
        write_json(config_file, cfg.as_dict())
        output = self.workspace / "tuning"
        calls = []

        def fake_suite(settings, output_dir, **kwargs):
            self.assertIs(kwargs.get("test"), False)
            self.assertEqual(settings.stage, "development")
            self.assertEqual(settings.cell_cohort, "development")
            calls.append((settings, output_dir))
            rates = settings.learning_rates[settings.conditions[0].learner]
            loss = abs(rates["lr_rec"] - 0.001) + 0.5
            return [{"best_validation_bce": loss} for _ in settings.seeds]

        with mock.patch.object(experiment, "run_suite", side_effect=fake_suite), redirect_stdout(io.StringIO()):
            status = cli_main(["tune", "--config", str(config_file), "--output", str(output)])
        self.assertEqual(status, 0)
        self.assertEqual(len(calls), 6)
        selected = json.loads((output / "selected_development_config.json").read_text())
        self.assertEqual(selected["learning_rates"]["bptt"]["lr_rec"], 0.001)
        self.assertEqual(selected["learning_rates"]["local"]["lr_rec"], 0.001)
        record = json.loads((output / "selection_record.json").read_text())
        self.assertFalse(record["test_evaluated"])

    def test_suite_aggregates_network_runs_and_marks_incomplete_factorial(self):
        cfg = self.config(conditions=[Condition("bptt"), Condition("local", "local")], seeds=(7, 11), training_steps=2, checkpoint_steps=(0, 2))
        output = self.workspace / "suite"
        with redirect_stdout(io.StringIO()):
            runs = experiment.run_suite(cfg, output, project_root=self.project)
        self.assertEqual(len(runs), 4)
        metrics = pd.read_csv(output / "metrics.csv")
        self.assertEqual(len(metrics), 4 * 5)
        self.assertEqual(set(metrics.seed), {7, 11})
        summaries = json.loads((output / "network_summary.json").read_text())
        self.assertTrue(all(row["n"] == 2 for row in summaries))
        contrasts = json.loads((output / "factorial_contrasts.json").read_text())
        self.assertEqual(contrasts["status"], "not_estimable_incomplete_factorial")
        complete = json.loads((output / "suite_completed.json").read_text())
        self.assertEqual(complete["n_runs"], 4)
        self.assertEqual(complete["statistical_unit"], "network_run_conditional_on_fitted_cell_resource")
        self.assertTrue((output / "behavior_summary.png").exists())
        self.assertTrue((output / "learning_curves.pdf").exists())

    def test_configuration_rejects_unreported_node_noise_and_ood_overlap(self):
        with self.assertRaises(ValueError):
            Condition("perturb", "node_perturbation", training_noise_std=0)
        with self.assertRaises(ValueError):
            self.config(ood_delays=(5,))
        with self.assertRaises(ValueError):
            self.config(checkpoint_steps=(0, 100))
        with self.assertRaises(ValueError):
            ExperimentConfig.from_dict({"unknown_typo": 1})


if __name__ == "__main__":
    unittest.main()
