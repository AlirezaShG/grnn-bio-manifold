# Executed development results

**Release evidence recorded on 7 November 2026.** This document summarizes the
saved executions bundled with version 7. These are development experiments on
the fitted-cell resource. They do not constitute independent physiological
validation, a confirmatory study or evidence of journal acceptance. All numbers
below come from the cited saved artifacts; no result is inferred from an
unexecuted configuration.

## What was actually executed

| Work item | Executions | Design and network seeds | Test evaluation | Evidence |
|---|---:|---|---|---|
| Initial factorial/control pilot | 45 | 15 conditions × seeds 42, 101, 202; includes the complete 2 × 2 × 2 factorial and seven additional controls | Five evaluation splits per run | [Completion record](../results/pilot/factorial/suite_completed.json), [configuration](../results/pilot/factorial/config.json), [training summary](../results/pilot/factorial/training_summary.csv) |
| Learning-rate search | 36 | Four methods × three candidates × seeds 42, 101, 202 | **Validation only**; test and mechanistic-analysis branches disabled | [Candidate results](../results/pilot/tuning/tuning_summary.json), [selection record](../results/pilot/tuning/selection_record.json) |
| Follow-up using selected rates | 12 | Four methods × fresh development seeds 303, 404, 505 | Five evaluation splits per run | [Completion record](../results/pilot/tuned/suite_completed.json), [configuration](../results/pilot/tuned/config.json), [training summary](../results/pilot/tuned/training_summary.csv) |
| Release verification | 110 tests | Numerical, unit, integration and data-integrity checks | Software verification, not additional experimental replication | [Validation record](../results/validation/validation.json), [test log](../results/validation/validation.log) |

There were **93 training executions** across the three experiment stages. This
is not a sample size of 93 independent replicates. Conditions and learning-rate
candidates share paired network seeds; the initial pilot and tuning search
also reuse seeds. Each displayed condition mean and each paired comparison
has **n = 3 network runs**. Hundreds of test trials, multiple cells and time
bins improve measurement within a run but do not enlarge that independent
network sample.

The follow-up uses new network seeds and corresponding fresh random streams,
but remains on the **development specimen cohort**. It must not be described
as use of the reserved confirmation cohort. The tested source fingerprint is
`5c447c992856f100270de8a0c19cb7e3b24d581122fd999978670874d3a1ab99`;
each suite and run manifest also preserves its configuration and input hashes.

## Shared task and training budget

The recorded pilots use 32 cells: 24 pyramidal, 4 PV, 2 SST and 2 VIP slots,
with authentic K=8 fitted neurons in the intact condition. Intrinsic neuron
parameters and input projections remain fixed. Every execution has 300
minibatch updates of 32 trials: **9,600 training trials per run**. The selected
checkpoint is the minimum-validation-BCE checkpoint, while the complete
training budget is still consumed.

The early-context task uses 20 ms bins, 100 ms fixation, 200 ms sensory
presentation, a silent delay sampled from 100/200/300 ms, and a 60 ms response
window. The delay-only OOD test uses 400/600 ms delays with the same sensory
noise as training, so its delay effect is separated from the sensory-noise,
distractor and combined-shift tests. Each evaluated split contains 256 trials
per network, including 128 conflict trials because the design is balanced.
The tuned follow-up uses 256 validation trials and 64 trials for each of its
independent analysis-fit and analysis-evaluation banks.

The equal-budget search considered three distinct rate pairs per method.
All four methods selected recurrent learning rate `0.003` and readout learning
rate `0.1` by mean per-seed best validation BCE. The recurrent rate has no
effect for the frozen-reservoir/readout-only method. The search is finite and
does not establish that any method is optimally tuned. Noisy BPTT and node
perturbation require their own matched-noise tuning study; their initial-pilot
results should not be presented as a tuned algorithm ranking. See the
[selection record](../results/pilot/tuning/selection_record.json).

## Four-method follow-up on fresh development seeds

Values are **mean ± sample standard deviation**, in percentage points, across
seeds **303, 404, 505**. Standard deviation uses `ddof=1`; it is neither a
standard error nor a confidence interval. “Conflict” means the two sensory
streams have opposite latent signs. It does not independently identify
distractor leakage as the cause of an error.

| Method | ID accuracy (%) | Delay-only OOD accuracy (%) | Delay-only OOD conflict accuracy (%) |
|---|---:|---:|---:|
| Exact network BPTT | 94.40 ± 3.93 | 71.61 ± 11.52 | 67.97 ± 5.63 |
| Local K-channel eligibility, including beta derivative | 82.81 ± 4.99 | 69.66 ± 14.12 | 61.20 ± 9.95 |
| Local eligibility with beta derivative removed | 73.05 ± 4.40 | 60.81 ± 5.30 | 47.92 ± 9.12 |
| Readout-only, frozen recurrent reservoir | 74.22 ± 3.77 | 61.07 ± 6.86 | 52.34 ± 5.47 |

Source: [tuned per-network metrics](../results/pilot/tuned/metrics.csv), using
`test_id/accuracy`, `test_delay/accuracy`, and
`test_delay/accuracy_incongruent`. [Network summaries](../results/pilot/tuned/network_summary.json)
retain estimates and bootstrap intervals separately. The individual network
points are displayed in [the behavior figure](../results/pilot/tuned/behavior_summary.png).

BPTT has the highest descriptive ID mean in this follow-up. On delay-only OOD,
its overall mean is only about 1.95 percentage points above the local rule,
with substantial variation across the three networks. The frozen reservoir's
nontrivial ID performance also shows why a recurrent-learning control is
necessary. These observations support further experiments; they do not
establish reliable superiority, equivalence or a biological mechanism.

Do not subtract the first pilot's means from these follow-up means and call the
difference a paired effect of tuning: the follow-up changed network seeds and
the initial pilot used a different validation-bank size. The actual
validation-only candidate comparison is recorded separately.

## Initial factorial family: no resolved primary effect

The 45-run pilot contains a complete factorial with intact versus within-type
shuffled identity, type-prior versus degree-rewired topology, and BPTT versus
local learning. Its declared primary outcome is **unseen-delay conflict
accuracy**. All seven factorial contrasts have **Holm-adjusted p = 1.0**.

The following effect estimates are mean ± sample SD across the **three paired
network seeds**, expressed in percentage points. Positive coding is intact
identity, type-prior topology and local learning. Main effects average over
the other factors; two-way effects are differences of differences, and the
three-way effect is a difference of differences of differences.

| Factorial term | Paired effect (percentage points) | Exact raw two-sided p | Holm p |
|---|---:|---:|---:|
| Identity | +1.63 ± 10.37 | 1.00 | 1.00 |
| Topology | −3.45 ± 1.86 | 0.25 | 1.00 |
| Learner: local minus BPTT | −10.48 ± 4.65 | 0.25 | 1.00 |
| Identity × topology | +8.72 ± 4.36 | 0.25 | 1.00 |
| Identity × learner | −6.38 ± 22.02 | 0.75 | 1.00 |
| Topology × learner | +0.13 ± 8.44 | 1.00 | 1.00 |
| Identity × topology × learner | +7.55 ± 17.75 | 0.75 | 1.00 |

Source: [saved factorial contrasts](../results/pilot/factorial/factorial_contrasts.json)
and [per-network metrics](../results/pilot/factorial/metrics.csv). With three
nonzero paired observations the smallest exact two-sided sign-flip p value is
`2 / 2**3 = 0.25`. The design therefore has very limited inferential resolution.
Adjusted p = 1.0 is not evidence that all effects are zero; the pilot simply
does not resolve the primary effects. The tuned four-method follow-up is
correctly marked as an incomplete factorial and supplies no replacement
factorial inference.

## A material operating-regime limitation of the identity control

Within-type shuffling preserves the marginal parameter distributions but
changes their combinations. In the executed factorial pilot, it also changes
the circuit's firing-rate and saturation regime substantially. The table below
uses **unseen-delay test trials, averaged over all neurons and all valid bins
of each whole trial**. These are **not delay-epoch-specific rates**.

Each entry is mean ± sample SD over the three network runs. Saturation is the
percentage of those valid neuron-time samples at or above 95% of that neuron's
fitted maximum rate.

| Learner / topology | Intact mean rate (Hz) | Shuffled mean rate (Hz) | Intact near-ceiling samples (%) | Shuffled near-ceiling samples (%) |
|---|---:|---:|---:|---:|
| BPTT / type prior | 13.65 ± 1.90 | 25.84 ± 5.65 | 0.41 ± 0.33 | 8.01 ± 3.11 |
| Local / type prior | 13.54 ± 2.50 | 26.14 ± 4.16 | 0.18 ± 0.23 | 7.14 ± 1.96 |
| BPTT / degree rewired | 14.33 ± 2.84 | 26.81 ± 5.19 | 0.40 ± 0.25 | 8.34 ± 3.97 |
| Local / degree rewired | 13.57 ± 2.80 | 27.32 ± 5.11 | 0.26 ± 0.36 | 9.14 ± 2.04 |

Source: [factorial metrics](../results/pilot/factorial/metrics.csv), filtered to
`split == "test_delay"`, using `mean_rate_hz` and `ceiling_rate_fraction`.

Thus the identity manipulation jointly changes within-cell covariance and the
resulting operating regime: approximately 26–27 Hz for shuffled circuits
versus 13–14 Hz for intact circuits, with higher saturation. Keeping the
external calibration fixed makes the intervention explicit, but does not
produce a rate-matched mechanistic comparison. Performance differences cannot
be attributed specifically to biologically meaningful covariance without
additional declared operating-point and rate-matched sensitivity analyses.
Any alternative calibration should be an explicitly separate control with
label-free selection, not a silent adjustment made only to rescue one result.

## Verification and what remains unexecuted

The saved release validation reports **110 tests, zero failures, zero errors
and zero skips**. Its finite-difference audit on actual Allen cells checks 33
recurrent edges, including 26 nonzero sensitivities, and records relative L2
error approximately `7.49e-10`. This is evidence for the implemented equations
and tested numerical contracts. It does not establish physiological validity
or general correctness beyond those checks. Raw held-out patch-clamp sweep
revalidation was **not performed**; those experimental sweeps are not bundled.

`configs/confirmation_template.json` is an **unexecuted 96-run template**:
eight factorial cells × 12 paired network seeds, 100 neurons, 1,500 updates per
run, and the reserved confirmation specimen cohort. No confirmation results
are included. The configuration is not an external preregistration, and its
sample size is not a completed power calculation. Before a confirmatory use,
lock the development-selected settings and hypotheses, plan precision/power,
address the operating-regime limitation, and reserve fresh tests. The reserved
cohort is specimen-disjoint, not automatically donor-disjoint.

The late-context and SGD sensitivity configurations and the optional
fixed-point solver are capabilities for subsequent studies. Their availability
and unit-test coverage must not be presented as completed replicated
scientific experiments in this release.

## Raw evidence map and reproduction

| Evidence | Location |
|---|---|
| Initial pilot configuration, all 45 outcomes, seven primary contrasts | `results/pilot/factorial/config.json`, `metrics.csv`, `factorial_contrasts.json` |
| All 36 validation-only candidate executions | `results/pilot/tuning/<learner>/candidate_<index>/runs/<condition>/<seed>/` |
| Rate-selection criterion and outcomes | `results/pilot/tuning/selection_record.json`, `tuning_summary.json` |
| Fresh-seed follow-up configuration and 12 outcomes | `configs/tuned_pilot.json`, `results/pilot/tuned/config.json`, `results/pilot/tuned/metrics.csv` |
| Per-trial task descriptors, targets, decisions and test losses | `results/pilot/{factorial,tuned}/runs/<condition>/<seed>/trial_predictions.csv` |
| Exact construction, source/input hashes and sampled cells | Per-run `manifest.json`, `sampled_cells.csv`, plus suite `data_provenance.json` |
| Selected/last model and optimizer state | Per-run `best_checkpoint.npz`, `last_checkpoint.npz` |
| Learning and checkpoint selection | Per-run `history.json`, `learning_curve.csv`, `completed.json` |
| Geometry, temporal decoding and saved analysis trajectories | Per-run `population_analysis.json`, `temporal_generalization.json`, `analysis_trajectories.npz` |
| Intervention, pulse and psychometric outputs | Per-run `rate_interventions.csv`, `rate_intervention_trials.csv`, `pulse_recovery.json`, `psychometrics.csv` |
| Software/equation validation | `results/validation/validation.json`, `validation.log` |

Reproduce the selected-rate follow-up in a new output directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m grnn_bio run \
  --config configs/tuned_pilot.json --output results/my_tuned_pilot
```

The complete scientific interpretation and prospective study requirements are
in [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md). The implementation is an
audited basis for further research; the supplied development outcomes do not
justify a biology-proof or journal-ready claim.
